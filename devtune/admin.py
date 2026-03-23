from django.contrib import admin
from .models import Chat,Category, KnowledgeFile
# Register your models here.
admin.site.register(Chat)
admin.site.register(Category)
admin.site.register(KnowledgeFile)