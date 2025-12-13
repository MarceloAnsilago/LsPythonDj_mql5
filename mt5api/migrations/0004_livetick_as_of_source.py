from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mt5api", "0003_dailypricepivot"),
    ]

    operations = [
        migrations.AddField(
            model_name="livetick",
            name="as_of",
            field=models.DateTimeField(blank=True, null=True, db_index=True),
        ),
        migrations.AddField(
            model_name="livetick",
            name="source",
            field=models.CharField(default="mt5", max_length=16, db_index=True),
        ),
    ]
