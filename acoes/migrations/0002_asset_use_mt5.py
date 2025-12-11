from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("acoes", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="asset",
            name="use_mt5",
            field=models.BooleanField(default=False),
        ),
    ]
