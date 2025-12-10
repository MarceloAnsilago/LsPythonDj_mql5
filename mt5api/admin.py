from django.contrib import admin

from .models import DailyPrice, LiveTick


@admin.register(LiveTick)
class LiveTickAdmin(admin.ModelAdmin):
    list_display = ("ticker", "timestamp", "bid", "ask", "last")
    list_filter = ("ticker",)
    search_fields = ("ticker",)
    ordering = ("-timestamp",)


@admin.register(DailyPrice)
class DailyPriceAdmin(admin.ModelAdmin):
    list_display = ("ticker", "date", "open", "high", "low", "close")
    list_filter = ("ticker", "date")
    search_fields = ("ticker",)
    ordering = ("-date", "ticker")
