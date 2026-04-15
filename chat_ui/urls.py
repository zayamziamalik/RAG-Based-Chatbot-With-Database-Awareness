from django.urls import path

from . import views


urlpatterns = [
    path("", views.chat_page, name="chat_page"),
    path("ask/", views.ask_question, name="ask_question"),
    path("refresh/", views.refresh_knowledge, name="refresh_knowledge"),
    path("clear/", views.clear_chat, name="clear_chat"),
]
