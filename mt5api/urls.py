from django.urls import path

from . import views

urlpatterns = [
    path("stream-assets/", views.stream_assets, name="mt5_stream_assets"),
    path("push-live-quote/", views.push_live_quote, name="push_live_quote"),
    path("get-signal/", views.get_signal, name="get_signal"),
]
