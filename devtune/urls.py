from django.urls import path
from . import views

app_name = 'devtune'

urlpatterns = [
    # Home with auto-created temp chat
    path('', views.home_view, name="home_view"),

    # Create new chat (category optional)
    path('create-chat/', views.create_chat, name="create_chat"),
    path('create-chat/<slug:category_slug>/', views.create_chat, name="create_chat"),

    # New chat (creates temp session)
    path('new/', views.new_chat, name="new_chat"),

    # Chat panel for viewing old chats
    path('chat/<slug:slug>/', views.chat_panel, name="chat_panel"),

    # AJAX - send message to FastAPI
    path('chat/<slug:slug>/send/', views.send_message_ajax, name="send_message_ajax"),
    
    # Delete chat
    path('chat/<slug:slug>/delete/', views.delete_chat, name="delete_chat"),
    # RAG endpoints: upload, index, list, delete
    path('chat/<slug:slug>/rag/upload/', views.rag_upload, name="rag_upload"),
    path('chat/<slug:slug>/rag/index/', views.rag_index, name="rag_index"),
    path('chat/<slug:slug>/rag/list/', views.rag_list, name="rag_list"),
    path('chat/<slug:slug>/rag/delete/', views.rag_delete, name="rag_delete"),
    path('chat/<slug:slug>/rag/search/', views.rag_search, name='rag_search'),
]