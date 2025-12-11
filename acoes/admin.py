from django.contrib import admin
from .models import Asset, UserAsset

@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    list_display = ('ticker','name','ticker_yf','is_active','use_mt5','logo_prefix')
    search_fields = ('ticker','name','ticker_yf')
    list_filter  = ('is_active','use_mt5')

@admin.register(UserAsset)
class UserAssetAdmin(admin.ModelAdmin):
    list_display = ('user','asset','created_at')
    search_fields = ('user__username','asset__ticker')
