import uuid

from django.conf import settings
from django.http import JsonResponse
from .utils import router, img_preprocess


@router('result', name='result')
def get_result(request):
    try:
        img1 = img_preprocess(request.FILES['img1'])
        img2 = img_preprocess(request.FILES['img2'])
        # 跑模型

        # 返回
        result_id = uuid.uuid4()
        return JsonResponse({'status': 'ok', 'result': settings.DOMAIN + settings.MEDIA_URL + f"{result_id}.png"})
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)})
