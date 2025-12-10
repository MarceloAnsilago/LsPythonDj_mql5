from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="DailyPrice",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ticker", models.CharField(max_length=10)),
                ("date", models.DateField()),
                ("open", models.FloatField()),
                ("high", models.FloatField()),
                ("low", models.FloatField()),
                ("close", models.FloatField()),
            ],
            options={
                "ordering": ["-date", "ticker"],
                "unique_together": {("ticker", "date")},
                "indexes": [models.Index(fields=["ticker", "date"], name="mt5api_dail_ticker__20c920_idx")],
            },
        ),
        migrations.CreateModel(
            name="LiveTick",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("ticker", models.CharField(db_index=True, max_length=10)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                ("bid", models.FloatField(blank=True, null=True)),
                ("ask", models.FloatField(blank=True, null=True)),
                ("last", models.FloatField()),
            ],
            options={
                "ordering": ["-timestamp"],
                "indexes": [models.Index(fields=["ticker", "-timestamp"], name="mt5api_live_ticker__61a163_idx")],
            },
        ),
    ]
