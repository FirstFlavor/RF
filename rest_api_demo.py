#!/usr/bin/python
# -*- coding: utf-8 -*-
import requests

#是否使用输入
#history_data= input('请输入history_data：')
data={'history_data':'Please input you resource_id and history_data in here'}
req = requests.post(url='http://api.pigbenhouse.cn/do_predict',data=data)
print(req.json())