#!/usr/bin/env python
# coding: utf-8

# # [Project 4]_유가_데이터를_활용한_주유소_시장분석

# ---

# ## 프로젝트 목표
# - <한국석유공사 제품별 주유소 판매가격> 데이터의 가격/지역/브랜드/셀프여부를 분석하여 주유소 시장 관련 인사이트 도출
# - 실제 연구/실무에서 활용되는 필수 분석 과정 및 기법에 대해 학습

# ---

# ## 프로젝트 목차
# 1. **데이터 Cleansing 및 Feature Engineering:** 분석을 위한 사전 점검 및 데이터 개괄 이해 <br>
#     1.1. 2018년 데이터 기준 데이터 Cleansing 및 Feature Engineering<br>
#     1.2. Cleansing 및 Feature Engineering 함수 생성 및 전체 년도 데이터 적용 <br>
#     1.3. 연도별 데이터 Outer Join<br>
# <br> 
# 
# 2. **주유소 개폐업 현황 분석:** 연도별 주유소 ID 비교를 통한 개폐업 현황 분석<br>
#     2.1. 연도별 개폐업 수치 분석<br>
# <br>
# 
# 3. **브랜드 분석:** 브랜드별 가격경쟁력 및 시장점유율 분석<br>
#     3.1. 주요 브랜드별 가격 Line Plot 분석<br>
#     3.2. 주요 브랜드별 지난 4년간 시장 점유율 Stacked Bar Plot 및 Heatmap 분석<br>
# <br>
# 
# 4. **가격 분석:** 주유소 및 지역 별 가격 편차 분석<br>
#     4.1. 가격 분포 Boxplot<br>
#     4.2. 지역별 가격 분포 Boxplot (Multiple Columns)<br>
# <br>
# 

# ---

# ## 데이터 출처
# -  https://www.data.go.kr/data/15044628/fileData.do
# - Opinet 유가내려받기: 2018 ~ 2021년 4개년에 대해 각각 6월 1일~7일 데이터 추출
# - 프로젝트에 필요한 컬럼만 추출

# ---

# ## 프로젝트 개요
# 
# 행정안전부 공공데이터 포털에 등재되어있는 `한국석유공사 제품별 주유소 판매가격`은 전국 10000개 이상의 주유소에 대해 가격/지역/브랜드/셀프여부 등 방대한 데이터를 제공하고 있습니다. 이 데이터를 정유업체 전략기획팀 실무자의 입장에서 분석하여 주유소 시장에 대한 인사이트를 도출해봅시다. 
# 
# 먼저, 주유소별로 7일치씩 쌓여있는 데이터를 요약하여 주유소별로 1개의 행이 되도록 각 년도 데이터를 가공 해봅시다. 그리고 이 데이터를 통해 지난 4년동안 몇개의 주유소가 개업 및 폐업 했는지 분석해 봅시다. 다음, 브랜드별 가격경쟁력 및 지난 4년간 시장 점유율 변화를 분석해 봅시다. 마지막으로 주유소별 가격 편차가 어느정도 되는지 알아보고, 지역별로도 유의미한 차이가 있는지 분석해 봅시다. 

# ## 1. 데이터 Cleansing 및 Feature Engineering

# 필요한 패키지를 `import`한 후 분석을 위한 사전 점검과 함께 데이터 개괄을 이해합니다.

# ### 1.1. 18년 데이터 기준 데이터 점검 및 Cleansing

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus']=False


# In[ ]:


f18 = pd.read_csv(f'./data/과거_판매가격(주유소)_2018.csv')


# In[ ]:


f18.head()


# In[ ]:


# 0번 row 제거
f18 = f18.drop(0) 


# In[ ]:


# 변수별 null값 확인 결과 null 없음
f18.isna().sum() 


# In[ ]:


# include='all': 카테고리형 변수도 정보 제공
f18.describe(include='all') 


# #### describe 점검 포인트:
# - unique 번호가 11673개이며 최대 7번까지 기록되었음
# - 기간이 수치로 인식되고 있음
# - unique 지역 개수가 229이어서 너무 많음
# - unique 상표 개수가 9개이므로 적절함
# - unique 셀프여부 개수가 2개이며, 셀프여부는 각각 절반정도 비중을 차지함
# - 휘발유 min이 0임

# In[ ]:


# 기간을 datetime 형태로 변환
f18['기간'] = f18['기간'].apply(lambda x:pd.to_datetime(str(int(x)))) 


# In[ ]:


# 지역 변수 중 첫 지역 구분만 컬럼 형성
region_len = f18['지역'].apply(lambda x: len(x.split())) 
print(f"min: {min(region_len)},max: {max(region_len)}")


# In[ ]:


f18['지역2'] = f18['지역'].apply(lambda x:x.split()[0])
import collections
collections.Counter(f18['지역2'])


# In[ ]:


# 휘발유값 0인 ROW 확인
f18.loc[f18['휘발유']==0].head(10) 


# In[ ]:


f18.loc[f18['번호']=='A0010629']


# In[ ]:


# 휘발유값 0인 ROW 제거
f18 = f18.loc[f18['휘발유']!=0,:]


# In[ ]:


f18.describe(include='all',datetime_is_numeric=True)


# In[ ]:


#주유소별 데이터 정합성 확인(7일동안 변화 없었다는 전제)
unique_count = f18.groupby('번호')[['지역','상표','셀프여부']].nunique()
unique_count.head()


# In[ ]:


target = unique_count.loc[(unique_count!=1).sum(axis=1)!=0]
target


# In[ ]:


f18.loc[f18['번호'].isin(target.index)]


# In[ ]:


f18 = f18.loc[~f18['번호'].isin(target.index)]


# In[ ]:


# 주유소별 데이터 통합
f18 = f18.groupby('번호')    .agg({'지역':'first','지역2':'first','상표':'first','셀프여부':'first','휘발유':'mean'})    .reset_index() 


# In[ ]:


f18.describe(include='all')


# ### 1.2. Cleansing 및 Feature Engineering 함수 생성 및 전체 년도 데이터 적용

# In[ ]:


def preprocess(df):
    df_copy=df.copy() # 필터링 전
    
    df = df.drop(0)
    df['기간'] = df['기간'].apply(lambda x:pd.to_datetime(str(int(x))))
    df['지역2'] = df['지역'].apply(lambda x:x.split()[0])
    df = df.loc[df['휘발유']!=0,:]
    unique_count = df.groupby('번호')[['번호','지역','상표','셀프여부']].nunique()
    target = unique_count.loc[(unique_count!=1).sum(axis=1)!=0,:]
    df = df.loc[~df['번호'].isin(target.index),:]
    df = df.groupby('번호')        .agg({'지역':'first','지역2':'first','상표':'first','셀프여부':'first','휘발유':'mean'})        .reset_index()
    
    out = set(df_copy['번호']).difference(set(df['번호'])) # 필터링 후 
    return(df,out)


# In[ ]:


f_dict = dict()
out_all = set() # 이상치 발견한 주유소 번호 저장
for year in range(2018,2022):
    df = pd.read_csv(f'./data/과거_판매가격(주유소)_{year}.csv')
    f_dict[year], out = preprocess(df)
    out_all.update(out)


# ### 1.3. 연도별 데이터 Outer Join

# In[ ]:


key = list(f_dict[2018].columns)
key.remove('휘발유')
print(key)


# In[ ]:


m1 = pd.merge(f_dict[2018],f_dict[2019],on=key,how='outer',suffixes=('_2018', '_2019'))
m2 = pd.merge(f_dict[2020],f_dict[2021],on=key,how='outer',suffixes=('_2020', '_2021'))
m = pd.merge(m1,m2,on=key,how='outer')


# In[ ]:


m.head()


# In[ ]:


m.groupby('번호').size().sort_values(ascending=False).head()


# In[ ]:


m.loc[m['번호']=='A0019752']


# In[ ]:


(m.groupby('번호').size()>1).sum()


# In[ ]:


key.remove('상표')
key


# In[ ]:


m1 = pd.merge(f_dict[2018],f_dict[2019],on=key,how='outer',suffixes=('_2018', '_2019'))
m2 = pd.merge(f_dict[2020],f_dict[2021],on=key,how='outer',suffixes=('_2020', '_2021'))
m = pd.merge(m1,m2,on=key,how='outer')


# In[ ]:


m.head()


# In[ ]:


size = m.groupby('번호').size().sort_values(ascending=False)
size.head()


# In[ ]:


target = size[size>1].index
m.loc[m['번호'].isin(target)].sort_values('번호')


# In[ ]:


m = m.loc[~m['번호'].isin(target)]
m.groupby('번호').size().sort_values(ascending=False).head()


# In[ ]:


# 이상치 발견되었던 주유소 필터링
m = m.loc[[x not in out_all for x in m['번호']]]


# In[ ]:


m.head()


# ---

# ## 2. 주유소 개폐업 현황 분석: 연도별 주유소 ID 비교를 통한 개폐업 현황 분석
# 

# ### 2.1. 연도별 개폐업 수치 분석

# In[ ]:


id_dict=dict()
for year in range(2018,2022):
    id_dict[year] = set(m.loc[~m[f'상표_{year}'].isna()]['번호'].unique())


# In[ ]:


diff_dict=dict()
for year in range(2018,2021):
    opened = len(id_dict[year+1].difference(id_dict[year]))
    closed = len(id_dict[year].difference(id_dict[year+1]))
    diff_dict[f'{year}_{year+1}']=[opened,closed]
diff_df = pd.DataFrame(diff_dict,index=['OPENED','CLOSED'])  


# In[ ]:


diff_df


# In[ ]:


diff_df.plot()


# In[ ]:


diff_df.T.plot(color=['r','b'])


# #### 퀴즈 1. 2020년에 신규 개업한 셀프 주유소의 개수를 구하시오.

# In[ ]:


# id_dict=dict()
# for year in range(2018,2022):
#     id_dict[year] = set(m.loc[(~m[f'상표_{year}'].isna())&(m['셀프여부']=='셀프')]['번호'].unique())
# diff_dict=dict()
# for year in range(2018,2021):
#     opened = len(id_dict[year+1].difference(id_dict[year]))
#     closed = len(id_dict[year].difference(id_dict[year+1]))
#     diff_dict[f'{year}_{year+1}']=[opened,closed]
# diff_df = pd.DataFrame(diff_dict,index=['OPENED','CLOSED'])    


# In[ ]:


#  2020년에 신규 개업한 셀프 주유소의 개수를 구하여 quiz_1 변수에 저장합니다.
# 숫자형으로 저장합니다.
quiz_1 = None


# ---

# ## 3. 브랜드 분석: 브랜드별 가격경쟁력 및 시장점유율 분석
# 

# ### 3.1. 주요 브랜드별 가격 Line Plot 분석

# In[ ]:


brand_price_dict=dict()
for year in range(2018,2022):
    brand_price_dict[str(year)]=m.groupby(f'상표_{year}')[f'휘발유_{year}'].mean()


# In[ ]:


brand_price_df = pd.DataFrame(brand_price_dict)
brand_price_df


# In[ ]:


brand_price_df = brand_price_df.drop('SK가스')
brand_price_df.T.plot(figsize=(10,5))


# ### 3.2. 주요 브랜드별 지난 4년간 시장 점유율 Stacked Bar Plot 및 Heatmap

# In[ ]:


brand_share_dict=dict()
for year in range(2018,2022):
    brand_share_dict[str(year)]=m.groupby(f'상표_{year}').size()


# In[ ]:


brand_share_df = pd.DataFrame(brand_share_dict)
brand_share_df


# In[ ]:


brand_share_df = brand_share_df.drop('SK가스')
brand_ratio_df = brand_share_df.apply(lambda x:x/brand_share_df.sum(),axis=1)
brand_ratio_df = brand_ratio_df.sort_values('2018',ascending=False)


# In[ ]:


brand_ratio_df


# In[ ]:


brand_ratio_df.T.plot(kind='bar',stacked=True,rot=0,figsize=(10,5))
plt.legend(bbox_to_anchor=(1, 1))


# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(brand_ratio_df, cmap= 'RdBu_r', linewidths=1, linecolor='black',annot=True)


# #### 퀴즈 2. 2019년 주유소를 셀프 및 일반 주유소로 구분하고 일반 주유소가 차지하는 비율을 구하시오

# In[ ]:


# self_share_dict = m.loc[~m['상표_2019'].isna()].groupby('셀프여부').size()
# self_ratio_dict = self_share_dict/self_share_dict.sum()


# In[ ]:


# 2019년 기준 셀프주유소의 시장 점유율을 quiz_2 변수에 저장합니다.
# 비율은 소숫점 둘째자리 까지 반올림하여 숫자형으로 제출합니다.
quiz_2 = None


# ---

# ## 4. **가격 분석:** 주유소 및 지역 별 가격 편차 분석

# ### 4.1. 가격 분포 Boxplot

# In[ ]:


sns.boxplot(x=m['휘발유_2021'])


# - Boxplot 설명:
# https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

# ### 4.2. 지역별 가격 분포 Boxplot (Multiple Columns)

# In[ ]:


boxplot_order = m.groupby('지역2')['휘발유_2021'].median().sort_values(ascending=False).index
plt.figure(figsize=(15,7))
sns.boxplot(x="지역2", y="휘발유_2021", data=m, orient="v", order=boxplot_order)


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
