# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from .classify import classify

# Create your views here.

@api_view(['POST'])
def getRating(request, to_train):
    sentence = request.POST.get('sentence')
    if to_train == 'train':
        rating = classify(sentence, 1)
        # rating = 1
    elif to_train == 'no_train':
        rating = classify(sentence, 0)
        # rating = 0
    return JsonResponse({'rating':str(rating)})
