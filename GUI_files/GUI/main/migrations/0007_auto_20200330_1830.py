# Generated by Django 3.0.3 on 2020-03-30 18:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0006_auto_20200330_1710'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='PatientDatabase',
            new_name='SessionDatabase',
        ),
        migrations.AlterModelOptions(
            name='sessiondatabase',
            options={'verbose_name_plural': 'Session Database'},
        ),
    ]
