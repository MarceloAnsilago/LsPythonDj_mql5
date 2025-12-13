from datetime import timedelta

import pandas as pd
from django.contrib import admin
from django.template.response import TemplateResponse
from django.urls import path
from django.utils import timezone
from django.utils.dateparse import parse_date

from cotacoes.models import QuoteDaily

from .models import DailyPrice, DailyPricePivot, LiveTick


@admin.register(LiveTick)
class LiveTickAdmin(admin.ModelAdmin):
    list_display = ("ticker", "timestamp", "as_of", "bid", "ask", "last", "source")
    list_filter = ("ticker",)
    search_fields = ("ticker",)
    ordering = ("-timestamp",)


@admin.register(DailyPrice)
class DailyPriceAdmin(admin.ModelAdmin):
    list_display = ("ticker", "date", "open", "high", "low", "close")
    list_filter = ("ticker", "date")
    search_fields = ("ticker",)
    ordering = ("-date", "ticker")


@admin.register(DailyPricePivot)
class DailyPricePivotAdmin(admin.ModelAdmin):
    change_list_template = "admin/mt5api/dailypricepivot/change_list.html"
    list_display = ()
    list_filter = ()
    search_fields = ()
    ordering = ("-date",)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path("", self.admin_site.admin_view(self.pivot_view), name="mt5api_dailypricepivot_changelist"),
        ]
        return custom + urls

    def pivot_view(self, request):
        ticker = (request.GET.get("ticker") or "").strip().upper()
        start_raw = (request.GET.get("start") or request.GET.get("start_date") or "").strip()
        end_raw = (request.GET.get("end") or request.GET.get("end_date") or "").strip()
        start_dt = parse_date(start_raw) if start_raw else None
        end_dt = parse_date(end_raw) if end_raw else None

        qs = QuoteDaily.objects.select_related("asset").all()
        if ticker:
            qs = qs.filter(asset__ticker__iexact=ticker)
        if start_dt:
            qs = qs.filter(date__gte=start_dt)
        elif not end_dt:
            default_start = timezone.localdate() - timedelta(days=120)
            qs = qs.filter(date__gte=default_start)
        if end_dt:
            qs = qs.filter(date__lte=end_dt)

        df = pd.DataFrame(list(qs.values("date", "asset__ticker", "close")))
        pivot_cols = []
        pivot_rows = []
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = df["asset__ticker"].str.upper()
            pivot = (
                df.pivot(index="date", columns="ticker", values="close")
                  .sort_index(ascending=False)
                  .round(4)
            )
            pivot_cols = list(pivot.columns)
            for idx, row in pivot.iterrows():
                pivot_rows.append(
                    {
                        "date": idx,
                        "values": [None if pd.isna(row[c]) else float(row[c]) for c in pivot_cols],
                    }
                )

        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "pivot_cols": pivot_cols,
            "pivot_rows": pivot_rows,
            "filter_ticker": ticker,
            "filter_start": start_raw,
            "filter_end": end_raw,
        }
        return TemplateResponse(request, self.change_list_template, context)
