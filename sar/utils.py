import io
from functools import wraps
from inspect import isclass
import numpy as np
import cv2
import scipy.io as sio
from PIL import Image
from django.urls import path
from django.views import View
from django.conf import settings


def router(route, name=""):
    def inner_warpper(func):
        from django.urls import get_resolver
        resolver = get_resolver()

        if isclass(func) and issubclass(func, View):
            resolver.url_patterns.append(path(f"{route}/", func.as_view(), name=name if name else None))
        else:
            resolver.url_patterns.append(path(f"{route}/", func, name=name if name else None))

        @wraps(func)
        def wrapper(request, *args, **kwargs):
            return func(request, *args, **kwargs)

        return wrapper

    return inner_warpper


def img_preprocess(img):
    return cv2.cvtColor(np.array(Image.open(io.BytesIO(img.read()))), cv2.COLOR_BGR2GRAY)


def convert_result_to_image(result, origin, output_filename):
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j] = 0 if result[i][j] == 1 else 255

    edges = cv2.Canny(result, 100, 300)

    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for i in range(color_edges.shape[0]):
        for j in range(color_edges.shape[1]):
            if np.all(color_edges[i, j] == [0, 0, 0]):
                color_edges[i, j] = origin[i, j]
            else:
                color_edges[i, j] = [0, 0, 255]
    RESULT_DIR = settings.STATIC_ROOT
    cv2.imwrite(RESULT_DIR / f"{output_filename}.png", color_edges)
