import io
import os.path
import threading
import time
import uuid

import cv2
import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.core.cache import cache

from sar.model import get_result
from sar.utils import router, img_preprocess, convert_result_to_image
from django.views import View


@router('sar', name='sar')
class ResultView(View):

    def run_model(self, im1, im2):
        result_id = uuid.uuid4()

        def run(im1, im2):
            lock = cache.get('lock', False)
            cache.set(result_id, '等待队列中...', timeout=None)
            while lock:
                lock = cache.get('lock', False)
                time.sleep(0.5)
            cache.set('lock', True, timeout=None)
            result = get_result(img_preprocess(im1), img_preprocess(im2), result_id)
            # cv2.imwrite(str(settings.RESULT_ROOT / f"{result_id}-result.png"), result)
            convert_result_to_image(result, [im1, im2], result_id)
            cache.set(result_id, 0, timeout=None)
            cache.set('lock', False, timeout=None)

        thread = threading.Thread(target=run, args=(im1, im2))
        thread.start()
        return result_id

    def get(self, request):
        id = request.GET.get('id')
        if not id:
            return JsonResponse({'status': 'error', 'msg': '请输入id'})
        if cache.get(id):
            return JsonResponse({'status': 'ok', 'msg': cache.get(id)})
        elif os.path.exists(settings.RESULT_ROOT / f"{id}-1.png"):
            return JsonResponse({'status': 'ok', 'msg': [settings.DOMAIN + settings.RESULT_URL + f"{id}-1.png",
                                                         settings.DOMAIN + settings.RESULT_URL + f"{id}-2.png"]})
        else:
            return JsonResponse({'status': 'error', 'msg': 'id不存在'})

    def post(self, request):
        try:
            im1 = np.array(Image.open(io.BytesIO(request.FILES['img1'].read())))
            im2 = np.array(Image.open(io.BytesIO(request.FILES['img2'].read())))
            result_id = self.run_model(im1, im2)
            return JsonResponse({'status': 'ok', 'id': result_id})
        except Exception as e:
            return JsonResponse({'status': 'error', 'msg': str(e)})
