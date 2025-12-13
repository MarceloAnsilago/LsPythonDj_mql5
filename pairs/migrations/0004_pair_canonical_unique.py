from __future__ import annotations

from django.db import migrations, models


def canonicalize_pairs(apps, schema_editor):
    Pair = apps.get_model("pairs", "Pair")
    seen: dict[tuple[int, int], int] = {}

    for pair in Pair.objects.all().order_by("id"):
        # Canonical order: left_id < right_id
        if pair.left_id and pair.right_id and pair.left_id > pair.right_id:
            pair.left_id, pair.right_id = pair.right_id, pair.left_id
            pair.save(update_fields=["left", "right"])

        key = (pair.left_id, pair.right_id)
        if key in seen:
            pair.delete()
        else:
            seen[key] = pair.id


class Migration(migrations.Migration):

    dependencies = [
        ("pairs", "0003_usermetricsconfig_half_life_max"),
    ]

    operations = [
        migrations.RunPython(canonicalize_pairs, migrations.RunPython.noop),
        migrations.AddIndex(
            model_name="pair",
            index=models.Index(fields=["left", "right"], name="pairs_pair_left_right_idx"),
        ),
        migrations.AddIndex(
            model_name="pair",
            index=models.Index(fields=["scan_cached_at"], name="pairs_pair_scan_cached_idx"),
        ),
        migrations.AddConstraint(
            model_name="pair",
            constraint=models.UniqueConstraint(fields=("left", "right"), name="pairs_pair_left_right_unique"),
        ),
    ]
