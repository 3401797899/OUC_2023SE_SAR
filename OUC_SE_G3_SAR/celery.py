import os
from celery import Celery

# 设置默认Django设置模块，Celery将会使用这个来找到Django项目
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'OUC_SE_G3_SAR.settings')

app = Celery('OUC_SE_G3_SAR', broker='redis://127.0.0.1:6379/2',
     backend='redis://127.0.0.1:6379/1')

# 自动发现在Django应用中的异步任务
app.autodiscover_tasks(['sar',])
