from django.db import models


# Create your models here.
class Logs(models.Model):
    im1 = models.CharField(max_length=200, verbose_name='图片1hash值')
    im2 = models.CharField(max_length=200, verbose_name='图片2hash值')
    im3 = models.CharField(max_length=200, verbose_name='图片3hash值')
    result_id = models.CharField(max_length=200, verbose_name='结果id')
    accuracy = models.FloatField(max_length=200, verbose_name='精确度')
