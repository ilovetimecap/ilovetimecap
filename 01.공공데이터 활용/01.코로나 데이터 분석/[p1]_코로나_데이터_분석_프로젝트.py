#!/usr/bin/env python
# coding: utf-8

# # [Project 1] 코로나 데이터 분석

# ---

# ## 프로젝트 목표
# - 서울시 코로나19 확진자 현황 데이터를 분석하여 유의미한 정보 도출
# - 탐색적 데이터 분석을 수행하기 위한 데이터 정제, 특성 엔지니어링, 시각화 방법 학습

# ---

# ## 프로젝트 목차
# 1. **데이터 읽기:** 코로나 데이터를 불러오고 Dataframe 구조를 확인<br>
#     1.1. 데이터 불러오기<br>
# <br> 
# 2. **데이터 정제:** 비어 있는 데이터 또는 쓸모 없는 데이터를 삭제<br>
#     2.1. 비어있는 column 지우기<br>
# <br>
# 3. **데이터 시각화:** 각 변수 별로 추가적인 정제 또는 feature engineering 과정을 거치고 시각화를 통하여 데이터의 특성 파악<br>
#     3.1. 확진일 데이터 전처리하기<br>
#     3.2. 월별 확진자 수 출력<br>
#     3.3. 8월 일별 확진자 수 출력<br>
#     3.4. 지역별 확진자 수 출력<br>
#     3.5. 8월달 지역별 확진자 수 출력<br>
#     3.6. 월별 관악구 확진자 수 출력<br>
#     3.7. 서울 지역에서 확진자를 지도에 출력<br>

# ---

# ## 데이터 출처
# -  https://www.data.go.kr/tcs/dss/selectFileDataDetailView.do?publicDataPk=15063273

# ---

# ## 프로젝트 개요
# 
# 2020년 초에 발생한 코로나19 바이러스는 세계적으로 대유행하였고 이에 대한 많은 분석이 이루어지고 있습니다. 유행 초기엔 이를 분석할 데이터가 충분하지 않았지만 6개월 이상 지난 지금은 다양한 데이터 기관에서 코로나 관련 데이터를 공공으로 제공하고 있습니다.
# 
# 이번 프로젝트에서는 국내 공공데이터 포털에서 제공하는 `서울시 코로나19 확진자 현황` 데이터를 바탕으로 탐색적 데이터 분석을 수행해보겠습니다. 국내 데이터 중 확진자 비율이 제일 높고 사람이 제일 많은 서울시의 데이터를 선정하였으며, 이를 바탕으로 코로나19의 확진 추이 및 환자 특성에 대해서 데이터를 바탕으로 알아봅시다.
# 
# 

# ---

#  

# ## 1. 데이터 읽기

# 필요한 패키지 설치 및 `import`한 후 `pandas`를 사용하여 데이터를 읽고 어떠한 데이터가 저장되어 있는지 확인합니다.

# ### 1.1. 데이터 불러오기

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# pd.read_csv를 통하여 dataframe 형태로 읽어옵니다.
corona_all=pd.read_csv("./data/서울시 코로나19 확진자 현황.csv")


# In[ ]:


# 상위 5개 데이터를 출력합니다.
corona_all.head()


# In[ ]:


# dataframe 정보를 요약하여 출력합니다. 
corona_all.info()


#  

# ---

# ## 2. 데이터 정제

# 데이터를 읽고 확인했다면 결측값(missing data), 이상치(outlier)를 처리하는 데이터 정제 과정을 수행하여 봅시다.

# ### 2.1. 비어있는 column 지우기

# `corona_all.info()` 코드를 통하여 `국적`, `환자정보`, `조치사항` 에 해당하는 데이터가 존재하지 않는 것을 알 수 있습니다.
# 
# `dataframe.drop()`를 사용하여 불필요한 `국적`, `환자정보`, `조치사항` 의 column 데이터를 삭제하고 이 dataframe을 `corona_del_col`에 저장해 봅시다.

# In[ ]:


# drop 함수를 사용하여 국적, 환자정보, 조치사항 coulmn 데이터를 삭제합니다.
corona_del_col = corona_all.drop(columns = ['국적','환자정보','조치사항'])


# In[ ]:


# 정제 처리된 dataframe 정보를 출력합니다.
corona_del_col.info()


# ---

# ## 3. 데이터 시각화

# 데이터 정제를 완료한 `corona_del_col` 데이터를 바탕으로 각 column의 변수별로 어떠한 데이터 분포를 하고 있는지 시각화를 통하여 알아봅시다.

# ### 3.1. 확진일 데이터 전처리하기

# `확진일` 데이터를 간단히 출력해보면 `월.일` 형태의 날짜 형식임을 알 수 있습니다.
# 
# 월별, 일별 분석을 위해서는 문자열 형식의 데이터를 나누어 숫자 형 데이터로 변환해 보겠습니다.

# In[ ]:


corona_del_col['확진일']


# #### `확진일` 데이터를 `month`, `day` 데이터로 나누기

# `확진일`에 저장된 문자열 데이터를 나누어 `month`, `day` column에 int64 형태로 저장해 봅시다.

# In[ ]:


# dataframe에 추가하기 전, 임시로 데이터를 저장해 둘 list를 선언합니다.
month = []
day = []

for data in corona_del_col['확진일']:
    # split 함수를 사용하여 월, 일을 나누어 list에 저장합니다.
    month.append(data.split('.')[0])
    day.append(data.split('.')[1])


# In[ ]:


# corona_del_col에 `month`, `day` column을 생성하며 동시에 list에 임시 저장된 데이터를 입력합니다.
corona_del_col['month'] = month
corona_del_col['day'] = day

corona_del_col['month'].astype('int64')
corona_del_col['day'].astype('int64')


#  

# ### 3.2. 월별 확진자 수 출력

# 나누어진 `month`의 데이터를 바탕으로 달별 확진자 수를 막대그래프로 출력해 보겠습니다.

# In[ ]:


# 그래프에서 x축의 순서를 정리하기 위하여 order list를 생성합니다.
order = []
for i in range(1,11):
    order.append(str(i))

order


# In[ ]:


# 그래프의 사이즈를 조절합니다.
plt.figure(figsize=(10,5))

# seaborn의 countplot 함수를 사용하여 출력합니다.
sns.set(style="darkgrid")
ax = sns.countplot(x="month", data=corona_del_col, palette="Set2", order = order)


# In[ ]:


# series의 plot 함수를 사용한 출력 방법도 있습니다.
corona_del_col['month'].value_counts().plot(kind='bar')


# In[ ]:


# value_counts()는 각 데이터를 세어서 내림차순으로 정리하는 함수입니다.
corona_del_col['month'].value_counts()


#  

# ### 3.3. 8월달 일별 확진자 수 출력

# 월별 확진자 수를 출력해보면 알 수 있듯이 8월에 확진자 수가 가장 많았습니다.
# 
# 이번엔 8월 동안 확진자 수가 어떻게 늘었는지 일별 확진자 수를 막대그래프로 출력해 봅시다.

# In[ ]:


# 그래프에서 x축의 순서를 정리하기 위하여 order list를 생성합니다.
order2 = []
for i in range(1,32):
    
    order2.append(str(i))

order2


# In[ ]:


# seaborn의 countplot 함수를 사용하여 출력합니다.
plt.figure(figsize=(20,10))
sns.set(style="darkgrid")
ax = sns.countplot(x="day", data=corona_del_col[corona_del_col['month'] == '8'], palette="rocket_r", order = order2)


# #### 퀴즈 1. 8월 평균 일별 확진자 수를 구하세요. (8월 총 확진자/31일)

# In[ ]:


# corona_del_col[corona_del_col['month'] == '8']['day'].count()/31
# corona_del_col[corona_del_col['month'] == '8']['day'].value_counts().mean()


# In[ ]:


# 8월 평균 확진자 수를 구하여 quiz_1 변수에 저장합니다.
# float 형 상수값으로 저장합니다.
quiz_1 = None


#  

# ### 3.4. 지역별 확진자 수 출력

# `지역` 데이터를 간단히 출력해보면 `oo구` 형태의 문자열 데이터임을 알 수 있습니다.

# In[ ]:


corona_del_col['지역']


# 이번에는 지역별로 확진자가 얼마나 있는지 막대그래프로 출력해 봅시다.

# In[ ]:


import matplotlib.font_manager as fm

font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)


# In[ ]:


plt.figure(figsize=(20,10))
# 한글 출력을 위해서 폰트 옵션을 설정합니다.
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="지역", data=corona_del_col, palette="Set2")


#  

# #### 지역 이상치 데이터 처리

# 위의 출력된 데이터를 보면 `종랑구`라는 잘못된 데이터와 `한국`이라는 지역과는 맞지 않는 데이터가 있음을 알 수 있습니다.
# 
# 기존 지역 데이터 특성에 맞도록 `종랑구` -> `중랑구`, `한국` -> `기타`로 데이터를 변경해 봅시다.

# In[ ]:


# replace 함수를 사용하여 해당 데이터를 변경합니다.
# 이상치가 처리된 데이터이기에 새로운 Dataframe으로 저장합니다.
corona_out_region = corona_del_col.replace({'종랑구':'중랑구', '한국':'기타'})


# In[ ]:


# 이상치가 처리된 데이터를 다시 출력해 봅시다.
plt.figure(figsize=(20,10))
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="지역", data=corona_out_region, palette="Set2")


#  

# ### 3.5. 8월달 지역별 확진자 수 출력

# 감염자가 많았던 8월에는 지역별로 확진자가 어떻게 분포되어 있는지 막대그래프로 출력해 봅시다.

# In[ ]:


# 논리연산을 이용한 조건을 다음과 같이 사용하면 해당 조건에 맞는 데이터를 출력할 수 있습니다.
corona_out_region[corona_del_col['month'] == '8']


# In[ ]:


# 그래프를 출력합니다.
plt.figure(figsize=(20,10))
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="지역", data=corona_out_region[corona_del_col['month'] == '8'], palette="Set2")


#  

# ### 3.6. 월별 관악구 확진자 수 출력

# 이번에는 확진자가 가장 많았던 관악구 내의 확진자 수가 월별로 어떻게 증가했는지 그 분포를 막대그래프로 출력해 봅시다.

# In[ ]:


# 해당 column을 지정하여 series 형태로 출력할 수 있습니다.
corona_out_region['month'][corona_out_region['지역'] == '관악구']


# In[ ]:


# 그래프를 출력합니다.
plt.figure(figsize=(10,5))
sns.set(style="darkgrid")
ax = sns.countplot(x="month", data=corona_out_region[corona_out_region['지역'] == '관악구'], palette="Set2", order = order)


#  

# ### 3.7. 서울 지역에서 확진자를 지도에 출력

# 지도를 출력하기 위한 라이브러리로 folium을 사용해 봅시다.

# In[ ]:


# 지도 출력을 위한 라이브러리 folium을 import 합니다.
import folium

# Map 함수를 사용하여 지도를 출력합니다.
map_osm = folium.Map(location=[37.529622, 126.984307], zoom_start=11)

map_osm


#  

# 지역마다 지도에 정보를 출력하기 위해서는 각 지역의 좌표정보가 필요합니다.
# 
# 이를 해결하기 위해서 서울시 행정구역 시군 정보 데이터를 불러와 사용합니다.
# 
# 데이터 출처: https://data.seoul.go.kr/dataList/OA-11677/S/1/datasetView.do

# In[ ]:


# CRS에 저장합니다.
CRS=pd.read_csv("./data/서울시 행정구역 시군구 정보 (좌표계_ WGS1984).csv")


# In[ ]:


# Dataframe을 출력해 봅니다.
CRS


#  

# 저장된 데이터에서 지역명이 서울의 중심지 `중구`인 데이터를 뽑아봅시다.

# In[ ]:


CRS[CRS['시군구명_한글'] == '중구']


#  

# 이제 for 문을 사용하여 지역마다 확진자를 원형 마커를 사용하여 지도에 출력해 봅시다.

# In[ ]:


# corona_out_region의 지역에는 'oo구' 이외로 `타시도`, `기타`에 해당되는 데이터가 존재 합니다.
# 위 데이터에 해당되는 위도, 경도를 찾을 수 없기에 삭제하여 corona_seoul로 저장합니다.
corona_seoul = corona_out_region.drop(corona_out_region[corona_out_region['지역'] == '타시도'].index)
corona_seoul = corona_seoul.drop(corona_out_region[corona_out_region['지역'] == '기타'].index)

# 서울 중심지 중구를 가운데 좌표로 잡아 지도를 출력합니다.
map_osm = folium.Map(location=[37.557945, 126.99419], zoom_start=11)

# 지역 정보를 set 함수를 사용하여 25개 고유의 지역을 뽑아냅니다.
for region in set(corona_seoul['지역']):

    # 해당 지역의 데이터 개수를 count에 저장합니다.
    count = len(corona_seoul[corona_seoul['지역'] == region])
    # 해당 지역의 데이터를 CRS에서 뽑아냅니다.
    CRS_region = CRS[CRS['시군구명_한글'] == region]

    # CircleMarker를 사용하여 지역마다 원형마커를 생성합니다.
    marker = folium.CircleMarker([CRS_region['위도'], CRS_region['경도']], # 위치
                                  radius=count/10 + 10,                 # 범위
                                  color='#3186cc',            # 선 색상
                                  fill_color='#3186cc',       # 면 색상
                                  popup=' '.join((region, str(count), '명'))) # 팝업 설정
    
    # 생성한 원형마커를 지도에 추가합니다.
    marker.add_to(map_osm)

map_osm


# #### 퀴즈 2. 6월에 확진자가 가장 많이 나온 지역을 구하세요.

# In[ ]:


# corona_out_region[corona_del_col['month'] == '6']['지역'].value_counts()


# In[ ]:


# 6월에 확진자가 가장 많이 나온 지역을 구하여 quiz_2 변수에 저장합니다.
# 문자형으로 저장합니다.
quiz_2 = None


#  ---

# ## 제출하기

# 퀴즈 1번과 2번을 수행 후, 아래 코드를 실행하면 `quiz_1 ~ 2` 변수가 저장된 csv 파일을 제작하여 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


d = {'quiz_1': [quiz_1], 'quiz_2': [quiz_2]}
df_quiz = pd.DataFrame(data=d)
df_quiz.to_csv("submission.csv",index=False)


# In[ ]:


answer=pd.read_csv('submission.csv')
answer.loc[0]['quiz_2']


# In[ ]:


# 채점을 수행하기 위하여 로그인
import sys
sys.path.append('vendor')
from elice_challenge import check_score, upload


# In[ ]:


# 제출 파일 업로드
await upload()


# In[ ]:


# 채점 수행
await check_score()


# ---

# <span style="color:rgb(120, 120, 120)">본 학습 자료를 포함한 사이트 내 모든 자료의 저작권은 엘리스에 있으며 외부로의 무단 복제, 배포 및 전송을 불허합니다.
# 
# Copyright @ elice all rights reserved</span>
