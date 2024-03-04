import hashlib
import io
import os.path
import uuid
import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from sar.model import get_result, get_accuracy
from sar.utils import router, img_preprocess, convert_result_to_image
from django.views import View
from celery import shared_task
import redis
import cv2
from .models import Logs

cache = redis.Redis(host='localhost', port=6379, db=7)


@shared_task
def process_images(im1, im2, result_id):
    im1 = np.array(im1).astype(np.uint8)
    im2 = np.array(im2).astype(np.uint8)
    result = get_result(im1, im2, result_id, cache)
    convert_result_to_image(result, [im1, im2], result_id)
    cache.delete(result_id)


@router('sar', name='sar')
class ResultView(View):

    def get(self, request):
        id = request.GET.get('id')
        if not id:
            return JsonResponse({'status': 'error', 'msg': '请输入id'})
        elif cache.get(id) is not None:
            return JsonResponse({'status': 'ok', 'msg': int(cache.get(id)) if cache.get(id).decode(
                encoding='utf-8').isdigit() else cache.get(id).decode(encoding='utf-8')})
        elif os.path.exists(settings.RESULT_ROOT / f"{id}-1.png"):
            return JsonResponse({'status': 'ok', 'msg': [settings.DOMAIN + settings.RESULT_URL + f"{id}-1.png",
                                                         settings.DOMAIN + settings.RESULT_URL + f"{id}-2.png"]})
        else:
            return JsonResponse({'status': 'error', 'msg': 'id不存在'})

    def post(self, request):
        try:
        # md5 查询是否已经识别过
            file1 = request.FILES['img1'].read()
            file2 = request.FILES['img2'].read()
            hash1 = hashlib.md5(file1)
            hash2 = hashlib.md5(file2)
            log = Logs.objects.filter(im1=hash1.hexdigest(), im2=hash2.hexdigest())
            if log.exists():
                log = log.first()
                return JsonResponse({'status': 'ok', 'id': log.result_id})
            def process_img(file):
                im1 = np.array(Image.open(io.BytesIO(file))).astype(np.uint8)
                if len(im1.shape) == 3 and im1.shape[2] == 3:  # 检查是否为三通道彩色图像
                    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                return im1
            result_id = str(uuid.uuid4())
            im1 = process_img(file1)
            im2 = process_img(file2)
            cache.set(result_id, '等待队列中...')
            process_images.delay(im1.tolist(), im2.tolist(), result_id)
            return JsonResponse({'status': 'ok', 'id': result_id})
        except Exception as e:
            return JsonResponse({'status': 'error', 'msg': str(e)})


@router('accuracy')
def get_accuracy_view(request):
    def process_img(name):
        im1 = np.array(Image.open(io.BytesIO(request.FILES[name].read()))).astype(np.uint8)
        if len(im1.shape) == 3 and im1.shape[2] == 3:  # 检查是否为三通道彩色图像
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        return im1

    im1 = process_img('img1')
    im2 = process_img('img2')
    im_gt = process_img('img_gt')
    im_gt = np.where(im_gt == 0, 1, 2)
    accuracy = get_accuracy(im1, im2, im_gt)
    return JsonResponse({'status': 'ok', 'msg': f'{accuracy:.2f}%'})
