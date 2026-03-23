from django.db import models
from django.utils.text import slugify
from django.utils.translation import gettext_lazy as _
from django.utils.timezone import now
from django.contrib.auth.models import User
import os
import requests
import logging
from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver
import random
import string

def generate_random_string(length: int=12):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def custom_upload_to(instance, filename):
    ext = filename.split('.')[-1]
    new_filename = f"{instance.Category_name}_{now().strftime('%Y-%m-%d_%H-%M-%S')}.{ext}"
    return os.path.join('Chat_pics', str(instance.Category_name), new_filename)

def arabic_slugify(str):
    str = str.replace(" ", "-")
    str = str.replace(",", "-")
    str = str.replace("(", "-")
    str = str.replace(")", "")
    str = str.replace("?", "")
    return str

class Chat(models.Model):
    Chat_session_id = models.CharField(max_length=30, blank=True, null=True, unique=True, verbose_name="Chat Name")
    Chat_owner = models.ForeignKey(User, blank=True, null=True, verbose_name=_("Chat Owner"), on_delete=models.CASCADE, related_name='chat_owner')
    Chat_Active = models.BooleanField(_("Chat Active"), default=False, help_text=_("Is this Chat Active?"))
    Chat_slug = models.SlugField(blank=True, null=True, unique=True)
    Chat_category = models.ForeignKey('Category', on_delete=models.CASCADE, blank=True, null=True)
    Chat_createdat = models.DateTimeField(_("Chat Created At"), auto_now_add=True)
    Chat_utility_params = models.JSONField(default=dict, null=True, blank=True)
    Chat_is_temporary = models.BooleanField(_("Is Temporary"), default=False, help_text=_("Temporary chat until first message"))
    Chat_title = models.CharField(max_length=200, blank=True, null=True, verbose_name="Chat Title")
    
    class Meta:
        verbose_name = _("Chat")
        verbose_name_plural = _("Chat")
        ordering = ['-Chat_createdat']

    def __str__(self):
        return self.Chat_session_id or "Unnamed Chat"
    
    def save(self, *args, **kwargs):
        if not self.Chat_session_id:
            self.Chat_session_id = self.Chat_owner.username + '_' + generate_random_string()
        if not self.Chat_slug:
            self.Chat_slug = slugify(self.Chat_session_id)
            if not self.Chat_slug:
                self.Chat_slug = arabic_slugify(self.Chat_session_id)
        super(Chat, self).save(*args, **kwargs)

    def get_display_title(self):
        """Return a user-friendly title for the chat"""
        if self.Chat_title:
            return self.Chat_title
        if self.Chat_is_temporary:
            return "New Chat"
        return f"{self.Chat_category.Category_name if self.Chat_category else 'Chat'}"


class Category(models.Model):
    Category_name = models.CharField(max_length=100, verbose_name="Category Name")    
    Category_slug = models.SlugField(blank=True, null=True, unique=True)
    Category_img = models.ImageField(upload_to=custom_upload_to, blank=True, null=True)
    Category_description = models.TextField(blank=True, null=True, verbose_name="Category Description")
    Category_is_default = models.BooleanField(default=False, verbose_name="Default Category")
    Category_meta_data = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        verbose_name = _("Category")
        verbose_name_plural = _("Categories")

    def save(self, *args, **kwargs):
        if not self.Category_slug:
            self.Category_slug = slugify(self.Category_name)
            if not self.Category_slug:
                self.Category_slug = arabic_slugify(self.Category_name)
        super(Category, self).save(*args, **kwargs)

    def __str__(self):
        return self.Category_name


class KnowledgeFile(models.Model):
    """Track user-uploaded KB files and index status"""
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='knowledge_files')
    file_name = models.CharField(max_length=300)
    file_path = models.CharField(max_length=1024, blank=True, null=True)
    doc_hash = models.CharField(max_length=128, blank=True, null=True)
    indexed = models.BooleanField(default=False)
    doc_ids = models.JSONField(default=list, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Knowledge File'
        verbose_name_plural = 'Knowledge Files'

    def __str__(self):
        return f"{self.owner.username}: {self.file_name}"

    # Note: cleanup on deletion (file removal and API deletion) is handled by post_delete signal below.


@receiver(post_delete, sender=KnowledgeFile)
def on_knowledgefile_delete(sender, instance, **kwargs):
    logger = logging.getLogger(__name__)
    api_base = os.environ.get('RAG_API_BASE', 'http://localhost:5555/DevTune/rag')
    try:
        payload = {
            'username': instance.owner.username,
            'session_id': '',
            'file_name': instance.file_name
        }
        try:
            res = requests.post(f"{api_base}/delete", json=payload, timeout=30)
            if res.status_code == 200:
                logger.info(f"Requested API deletion of file {instance.file_name} for user {instance.owner.username} (post_delete)")
            else:
                logger.warning(f"API deletion returned status {res.status_code}: {res.text}")
        except Exception as e:
            logger.error(f"Error calling API to delete file {instance.file_name}: {e}")
        # Remove file from disk if present
        if instance.file_path and os.path.exists(instance.file_path):
            try:
                os.remove(instance.file_path)
                logger.info(f"Removed file from disk: {instance.file_path} (post_delete)")
            except Exception as e:
                logger.error(f"Error removing file {instance.file_path} from disk: {e}")
    except Exception as e:
        logger.error(f"Error during on_knowledgefile_delete: {e}")
