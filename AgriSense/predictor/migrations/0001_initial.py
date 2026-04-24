from django.db import migrations, models
import django.utils.timezone

class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SoilRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('n', models.FloatField()),
                ('p', models.FloatField()),
                ('k', models.FloatField()),
                ('ph', models.FloatField()),
                ('temperature', models.FloatField()),
                ('humidity', models.FloatField()),
                ('rainfall', models.FloatField()),
                ('location', models.CharField(blank=True, max_length=255)),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('recommended_crop', models.CharField(max_length=100)),
                ('yield_estimate', models.FloatField()),
                ('confidence_score', models.FloatField()),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('soil_record', models.ForeignKey(on_delete=models.deletion.CASCADE, to='predictor.soilrecord')),
            ],
        ),
    ]
