import io
import os.path
import uuid
import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from sar.model import get_result
from sar.utils import router, img_preprocess, convert_result_to_image
from django.views import View
from celery import shared_task
import redis

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
            result_id = str(uuid.uuid4())
            im1 = np.array(Image.open(io.BytesIO(request.FILES['img1'].read())))
            im2 = np.array(Image.open(io.BytesIO(request.FILES['img2'].read())))
            cache.set(result_id, '等待队列中...')
            process_images.delay(im1.tolist(), im2.tolist(), result_id)
            return JsonResponse({'status': 'ok', 'id': result_id})
        except Exception as e:
            return JsonResponse({'status': 'error', 'msg': str(e)})
