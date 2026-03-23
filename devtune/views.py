from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Chat, Category, KnowledgeFile
from django.http import JsonResponse
import os
import requests
import io
import json
from django.conf import settings
from django.db.models import Q

FASTAPI_BASE = "http://localhost:5555/DevTune/chat"
RAG_API_BASE = "http://localhost:5555/DevTune/rag"

def home_view(request):
    """
    Home view that auto-creates a temporary chat session for logged-in users
    """
    categories = Category.objects.all()
    user_chats = []
    current_chat = None
    
    if request.user.is_authenticated:
        # Get all user chats (excluding temporary ones)
        user_chats = Chat.objects.filter(
            Chat_owner=request.user,
            Chat_is_temporary=False
        ).order_by('-Chat_createdat')
        
        # Check if there's an active temporary chat
        temp_chat = Chat.objects.filter(
            Chat_owner=request.user,
            Chat_is_temporary=True
        ).first()
        
        if temp_chat:
            current_chat = temp_chat
        else:
            # Only create a new temp chat if there are no permanent chats OR explicitly requested
            # This prevents auto-creation when user just wants to browse their chats
            should_create_temp = request.GET.get('new') == '1' or user_chats.count() == 0
            
            if should_create_temp:
                # Create a new temporary chat with default category
                default_category = Category.objects.filter(Category_is_default=True).first()
                if not default_category:
                    default_category = Category.objects.first()
                
                utility_params=default_category.Category_meta_data
                current_chat = Chat.objects.create(
                    Chat_owner=request.user,
                    Chat_category=default_category,
                    Chat_Active=True,
                    Chat_is_temporary=True,
                    Chat_utility_params=utility_params
                )
            elif user_chats.exists():
                # If user has chats and no temp chat, show the most recent one
                current_chat = user_chats.first()
    
    return render(request, 'devtune/devtune_home.html', {
        "categories": categories,
        "user_chats": user_chats,
        "current_chat": current_chat,
    })


@login_required
def create_chat(request, category_slug=None):
    category = None
    utility_params = {}
    utility_params = {"completion_type": "main"}
    if category_slug:
        category = get_object_or_404(Category, Category_slug=category_slug)
        utility_params=category.Category_meta_data
    else:
        utility_params["completion_type"] = "chat"

    # Always create chat automatically
    Chat.objects.filter(
        Chat_owner=request.user,
        Chat_is_temporary=True
    ).delete()

    chat = Chat.objects.create(
        Chat_owner=request.user,
        Chat_category=category,
        Chat_Active=True,
        Chat_is_temporary=False,
        Chat_utility_params=utility_params
    )

    # Redirect to home with this chat
    return redirect(f"/?viewer={chat.Chat_slug}")

@login_required
def chat_panel(request, slug):
    """
    Display a specific chat (for old chats in sidebar)
    Redirects to home if viewing a temporary chat
    """
    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    
    # If this is a temporary chat, redirect to home view
    if chat.Chat_is_temporary:
        return redirect("devtune:home_view")
    
    user_chats = Chat.objects.filter(
        Chat_owner=request.user,
        Chat_is_temporary=False
    ).order_by('-Chat_createdat')
    
    return render(request, "devtune/chat_panel.html", {
        "chat": chat,
        "user_chats": user_chats,
        "current_chat": chat,
    })


@login_required
def send_message_ajax(request, slug):
    """
    AJAX endpoint that sends user message to FastAPI
    Converts temporary chat to permanent on first message
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    
    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    msg = request.POST.get("message", "").strip()
    
    if not msg:
        return JsonResponse({"error": "Empty message"}, status=400)
    
    # Track if this was a temporary chat
    was_temporary = chat.Chat_is_temporary
    
    # If this is the first message in a temporary chat, convert it to permanent
    if chat.Chat_is_temporary:
        chat.Chat_is_temporary = False
        # Generate a title from the first message (first 50 chars)
        chat.Chat_title = msg[:50] + ("..." if len(msg) > 50 else "")
        chat.Chat_utility_params["completion_type"] = "main"
        chat.save()
    
    use_rag_param = request.POST.get('use_rag', None)
    # Update utility_params copy with use_rag flag and optionally rag_top_k
    utility_params = chat.Chat_utility_params.copy() if chat.Chat_utility_params else {}
    if use_rag_param is not None:
        # Interpret use_rag string values '1', '0', 'true', 'false'
        val = str(use_rag_param).strip().lower()
        if val in ['1', 'true', 't', 'yes', 'y']:
            utility_params['use_rag'] = True
        else:
            utility_params['use_rag'] = False
        try:
            utility_params['rag_top_k'] = int(request.POST.get('rag_top_k', utility_params.get('rag_top_k', 3)))
        except Exception:
            utility_params['rag_top_k'] = utility_params.get('rag_top_k', 3)

    payload = {
        "username": request.user.username,
        "session_id": chat.Chat_session_id,
        "prompt": msg,
        "chat_history": [],
        "utility_params": utility_params
    }
    #print(f"[DevTune] Sending message to FastAPI: username={payload['username']}, session_id={payload['session_id']}, prompt={msg}, utility_params={utility_params}")
    
    try:
        res = requests.post(f"{FASTAPI_BASE}/complete", json=payload)

        if res.status_code == 200:
            return JsonResponse({
                **res.json(),
                "chat_converted": was_temporary,
                "chat_title": chat.Chat_title,
                "chat_slug": chat.Chat_slug
            })
        else:
            return JsonResponse({"error": "FastAPI error"}, status=500)
    except requests.exceptions.RequestException as e:
        print(str(e))
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)


@login_required
def rag_upload(request, slug):
    """
    Upload a document using the FastAPI RAG endpoint
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    file = request.FILES.get('file')
    if not file:
        return JsonResponse({"error": "No file provided"}, status=400)

    files = {
        # `file.file` is a file-like object; requests will stream it to the API with filename
        'file': (file.name, file.file, file.content_type)
    }
    data = {
        'username': request.user.username,
        'session_id': chat.Chat_session_id
    }
    try:
        # Force username to match currently logged-in user
        data['username'] = request.user.username
        print(f"[DevTune] Uploading to API: username={data.get('username')}, session_id={data.get('session_id')}, file_name={file.name}")
        res = requests.post(f"{RAG_API_BASE}/upload", files=files, data=data, timeout=60)
        resp_json = res.json()
        # If upload succeeded and indexing occurred, create KnowledgeFile record
        if res.status_code == 200 and resp_json.get('success'):
            file_name = resp_json.get('file_name') or resp_json.get('saved_file_name')
            file_path = resp_json.get('file_path')
            try:
                if file_path and os.path.exists(file_path):
                    print(f"[DevTune] Uploaded file path verified on API server: {file_path}")
                else:
                    print(f"[DevTune] Warning: uploaded file path not found: {file_path}")
            except Exception as e:
                print('[DevTune] Error checking file path existence:', e)
            validation = resp_json.get('validation') or {}
            index_result = resp_json.get('index_result') or {}
            doc_ids = index_result.get('doc_ids') or []
            doc_hash = validation.get('doc_hash') if isinstance(validation, dict) else None
            # Create or update KnowledgeFile entry
            try:
                if file_name:
                    kf, created = KnowledgeFile.objects.get_or_create(owner=request.user, file_name=file_name, defaults={'file_path': file_path, 'doc_hash': doc_hash, 'indexed': bool(doc_ids), 'doc_ids': doc_ids})
                else:
                    kf = None
                if not created:
                    kf.file_path = file_path
                    if doc_hash:
                        kf.doc_hash = doc_hash
                    kf.indexed = bool(doc_ids)
                    if doc_ids:
                        kf.doc_ids = doc_ids
                    kf.save()
            except Exception as e:
                # Log but continue, do not fail upload
                print('KnowledgeFile update failed: ', e)
        print('[DevTune] API upload response:', resp_json)
        return JsonResponse(resp_json, status=res.status_code)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)


@login_required
def rag_index(request, slug):
    """
    Index a document already uploaded
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    file_name = request.POST.get('file_name')
    utility_params = request.POST.get('utility_params', None)
    if not file_name:
        return JsonResponse({"error": "file_name is required"}, status=400)

    try:
        print(f"[DevTune] Index request: username={request.user.username}, file_name={file_name}, session_id={chat.Chat_session_id}")
        utility_params = json.loads(utility_params) if utility_params else chat.Chat_utility_params or {}
    except Exception:
        utility_params = chat.Chat_utility_params or {}

    payload = {
        'username': request.user.username,
        'session_id': chat.Chat_session_id,
        'file_name': file_name,
        'utility_params': utility_params
    }
    # If the DB knows the full file_path for this KnowledgeFile, prefer that
    try:
        kf = KnowledgeFile.objects.filter(owner=request.user, file_name=file_name).first()
        if kf and kf.file_path:
            payload['file_path'] = kf.file_path
    except Exception:
        pass
    try:
        res = requests.post(f"{RAG_API_BASE}/index", json=payload, timeout=120)
        resp_json = res.json()
        # Update KnowledgeFile model if indexing succeeded
        if res.status_code == 200 and resp_json.get('success'):
            try:
                kf = KnowledgeFile.objects.filter(owner=request.user, file_name=file_name).first()
                if kf:
                    kf.indexed = True
                    if resp_json.get('doc_ids'):
                        kf.doc_ids = resp_json.get('doc_ids')
                    kf.save()
            except Exception as e:
                print('KnowledgeFile update failed on index: ', e)
        return JsonResponse(resp_json, status=res.status_code)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)


@login_required
def rag_list(request, slug):
    """
    List documents in vector store for the current user
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    limit = int(request.POST.get('limit', 100))
    offset = int(request.POST.get('offset', 0))
    payload = {
        'username': request.user.username,
        'session_id': chat.Chat_session_id,
        'limit': limit,
        'offset': offset
    }
    try:
        # Use local KnowledgeFile DB for listing to return ownership info and index state
        kfs = KnowledgeFile.objects.filter(owner=request.user).order_by('-created_at')[offset:offset+limit]
        docs = []
        for k in kfs:
            docs.append({
                'file_name': k.file_name,
                'file_path': k.file_path,
                'doc_hash': k.doc_hash,
                'indexed': k.indexed,
                'doc_ids': k.doc_ids or [],
                'created_at': k.created_at.isoformat()
            })
        return JsonResponse({'success': True, 'documents': docs, 'total_count': KnowledgeFile.objects.filter(owner=request.user).count()})
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)


@login_required
def rag_delete(request, slug):
    """
    Delete a document from the vector store
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    doc_id = request.POST.get('doc_id')
    file_name = request.POST.get('file_name')
    if not doc_id and not file_name:
        return JsonResponse({"error": "doc_id or file_name is required"}, status=400)
    payload = {'username': request.user.username, 'session_id': chat.Chat_session_id}
    if doc_id:
        payload['doc_id'] = doc_id
    else:
        payload['file_name'] = file_name
    try:
        print(f"[DevTune] Delete called with payload: doc_id={doc_id}, file_name={file_name}, username={request.user.username}")
        if doc_id:
            # Delete by doc_id in vector store
            res = requests.post(f"{RAG_API_BASE}/delete", json=payload, timeout=30)
            resp_json = res.json()
        else:
            # Delete entire file: rely on model's post_delete handler to call API delete and remove file.
            kf = KnowledgeFile.objects.filter(owner=request.user, file_name=file_name).first()
            if kf:
                kf.delete()
                resp_json = {'success': True, 'message': 'File and its KB entries deleted locally'}
            else:
                resp_json = {'success': False, 'message': 'KnowledgeFile not found'}
        # If delete succeeded on API, try to update local KnowledgeFile record
        if (doc_id and res.status_code == 200 and resp_json.get('success')) or (not doc_id and resp_json.get('success')):
            try:
                # Find any KnowledgeFile containing this doc_id
                if doc_id:
                    kf = KnowledgeFile.objects.filter(owner=request.user, doc_ids__contains=[doc_id]).first()
                    if kf:
                        current_ids = kf.doc_ids or []
                        if doc_id in current_ids:
                            current_ids = [i for i in current_ids if i != doc_id]
                            kf.doc_ids = current_ids
                            if not current_ids:
                                kf.indexed = False
                            kf.save()
                elif file_name:
                    # Delete local KnowledgeFile record for this file
                    kf = KnowledgeFile.objects.filter(owner=request.user, file_name=file_name).first()
                    if kf:
                        # Optionally remove the uploaded file
                        try:
                            import os
                            if kf.file_path and os.path.exists(kf.file_path):
                                os.remove(kf.file_path)
                        except Exception:
                            pass
                        print(f"[DevTune] Deleting local KnowledgeFile record for owner={request.user.username}, file_name={file_name}")
                        kf.delete()
            except Exception as e:
                print('KnowledgeFile update failed on delete: ', e)
        if doc_id:
            return JsonResponse(resp_json, status=res.status_code)
        else:
            return JsonResponse(resp_json, status=200)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)





@login_required
def rag_search(request, slug):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
    query = request.POST.get('query', '')
    top_k = int(request.POST.get('top_k', 5))
    if not query:
        return JsonResponse({"error": "query is required"}, status=400)
    payload = {
        'username': request.user.username,
        'session_id': chat.Chat_session_id,
        'query': query,
        'top_k': top_k
    }
    try:
        res = requests.post(f"{RAG_API_BASE}/search", json=payload, timeout=30)
        return JsonResponse(res.json(), status=res.status_code)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)




@login_required
def delete_chat(request, slug):
    """
    Delete a chat session
    """
    if request.method == "POST":
        chat = get_object_or_404(Chat, Chat_slug=slug, Chat_owner=request.user)
        chat.delete()
        return JsonResponse({"success": True})
    return JsonResponse({"error": "Invalid request"}, status=400)


@login_required
def new_chat(request):
    """
    Create a new temporary chat and redirect to home
    """
    # Delete existing temporary chats
    Chat.objects.filter(
        Chat_owner=request.user,
        Chat_is_temporary=True
    ).delete()
    
    # Redirect to home with flag to create new temp chat
    return redirect("devtune:create_chat")