from django.urls import path
from . import views

urlpatterns = [
    path('',                          views.dashboard,   name='dashboard'),
    path('scan/<str:model_key>/',     views.scan_page,   name='scan_page'),
    path('scan/<str:model_key>/run/', views.run_scan,    name='run_scan'),
    path('result/<uuid:scan_id>/',    views.result_page, name='result_page'),
    path('history/',                  views.history,     name='history'),
    path('delete/<uuid:scan_id>/',    views.delete_scan, name='delete_scan'),
]
