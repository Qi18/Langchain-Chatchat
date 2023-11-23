import time

import jionlp as jio


time_text_list = ['前一个月习近平去了哪里', '从2018年12月九号到十五号', '2019年感恩节', '每周六上午9点到11点', '30~90日']
for time_text in time_text_list:
    print(jio.parse_time(time_text, time_base=time.time()))
try:
    time = jio.parse_time(query, time_base=time.time())
except Exception as e:
    with open("./timewords", "r") as file:
        words = file.readlines()
        for word in words:
            query = query.replace(word, "最近一个月")
            time = jio.parse_time(query, time_base=time.time())
            break
