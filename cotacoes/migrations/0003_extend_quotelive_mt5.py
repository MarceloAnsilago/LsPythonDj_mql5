from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cotacoes", "0002_quotedaily_is_provisional"),
    ]

    operations = [
        migrations.AddField(
            model_name="quotelive",
            name="ask",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="quotelive",
            name="as_of",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="quotelive",
            name="bid",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="quotelive",
            name="last",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="quotelive",
            name="source",
            field=models.CharField(default="mt5", max_length=32),
        ),
        migrations.AddField(
            model_name="quotelive",
            name="volume",
            field=models.BigIntegerField(blank=True, null=True),
        ),
    ]
