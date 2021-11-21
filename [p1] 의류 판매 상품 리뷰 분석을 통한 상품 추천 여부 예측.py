#!/usr/bin/env python
# coding: utf-8

# # [Project 1] 의류 판매 상품 리뷰 분석을 통한 상품 추천 여부 예측

# ---

# ## 프로젝트 목표
# - 상품 리뷰 데이터 분석을 통하여 상품 추천 여부를 예측하는 분류 모델 수행
# - 상품 추천 여부에 영향을 미치는 특성 데이터들에 대한 데이터 분석 수행

# ---

# ## 프로젝트 목차
# 
# 1. **데이터 읽기:** 이커머스 데이터를 불러오고 Dataframe 구조를 확인
# 
# 
# 2. **데이터 정제:** 비어 있는 데이터 또는 쓸모 없는 데이터를 삭제<br>
#     2.1. 결측값 확인<br>
#     2.2. 결측값 처리<br>
# 
# 
# 3. **데이터 시각화:** 변수 시각화를 통하여 분포 파악<br>
#     3.1. `Title` word cloud<br>
#     3.2. `Review Text` word cloud<br>
#     3.3. `Recommended IND` 시각화<br>
# 
# 
# 4. **데이터 전 처리:** 머신러닝 모델에 필요한 입력값 형식으로 데이터 처리<br>
#     4.1. 자연어 전 처리 - Tfidf<br>
#     4.2. 학습, 테스트 데이터 분리<br>
# 
# 
# 5. **머신러닝 모델 학습:** 분류 모델을 사용하여 학습 수행<br>
#     5.1. 기본 분류 모델 학습 - 의사결정나무<br>
#     5.2. 다양한 분류 모델 학습<br>
# 
# 
# 6. **평가 및 예측:** 학습된 모델을 바탕으로 평가 및 예측 수행<br>
#     6.1. Confusion Matrix<br>
#     6.2. Precision & Recall<br>
#     6.3. 테스트 데이터의 예측값 출력

# ---

# ## 데이터 출처 
# - https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

# ---

# ## 프로젝트 개요
# 
# 언택트 시대가 다가오면서 의류, 식료품, 전자제품 등 다양한 상품을 인터넷을 통하여 거래하게 되었습니다. 수많은 상품이 인터넷상에서 전시되어 있지만, 실제로 사용한 상품이 아니라면 나에게 맞는 상품인지를 알기 쉽지 않습니다. 이러한 단점을 극복하기 위하여 수많은 이커머스 웹사이트에서는 상품 리뷰 글을 남길 수 있도록 하고 최대한 많은 리뷰가 나올 수 있도록 장려하고 있습니다. 소비자는 이러한 리뷰를 통하여 실제 후기와 상품 정보에서 미쳐 파악하지 못한 정보들을 얻을 수 있으며, 긍정적인 리뷰가 많은 상품에 대해서는 신뢰도 또한 가질 수 있게 됩니다. 판매자 입장에서도 리뷰가 많은 상품을 마케팅 요소로 사용할 수 있으며, 소비자의 의견을 들을 수 있는 장점이 있습니다. 따라서 이커머스에서는 리뷰는 상품을 구매하는 데 있어서 매우 중요한 특성 데이터로서 가치가 있습니다.
# 
# 이번 프로젝트에서는 Kaggle에서 제공하는 여성 의류 이커머스 데이터를 바탕으로 상품 추천 여부를 예측해보는 분류 모델을 구현합니다. 이를 통하여 새로운 리뷰와 여성 의류 정보를 받았을 때, 해당 상품을 추천하는지를 예측할 수 있습니다. 이를 활용하면 상품 추천 여부가 없는 상품들에 대해서도 리뷰와 의류 정보를 바탕으로 상품 추천 여부를 예측할 수 있으며, 이는 상품의 중요한 지표로서 다시 활용될 수 있습니다. 

# ---

# In[ ]:


get_ipython().system('pip install wordcloud')


#  

# ## 1. 데이터 읽기

# pandas를 사용하여 `Womens Clothing E-Commerce Reviews(수정).csv` 데이터를 읽고 dataframe 형태로 저장해 봅시다.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Womens Clothing E-Commerce Reviews(수정).csv 데이터를 pandas를 사용하여 dataframe 형태로 불러옵니다.
df_origin = pd.read_csv("./data/Womens Clothing E-Commerce Reviews(수정).csv")


# In[ ]:


# 5개의 데이터 샘플을 출력합니다.
df_origin.head()


# In[ ]:


# dataframe의 정보를 요약해서 출력합니다.
df_origin.info()


# In[ ]:


# 수치형 변수의 데이터 정보를 요약하여 출력합니다.
df_origin.describe()


#  

# ## 2. 데이터 정제

# 일반적으로 데이터 정제에서는 **결측값(missing value)** 또는 **이상치(outlier)**를 처리합니다.
# 
# 이상치 데이터는 각 변수의 특성에 맞게 처리하는 것이 좋기에 각 변수를 살펴보는 데이터 시각화를 수행하며 필요하게 된다면 추가 적으로 수행하겠습니다.
# 
# 따라서 이번 파트에서는 간단하게 결측값에 대해서만 처리를 수행하겠습니다.

# In[ ]:


# 결측값을 처리하기 전에 우선 의미 없는 변수인 'Unnamed: 0, Unnamed: 0.1'를 drop을 사용하여 삭제합니다.
df_clean = df_origin.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])


# ### 2.1. 결측값 확인

# 아래 코드를 수행하여 각 변수별로 결측값이 몇개가 있는지 확인합니다.

# In[ ]:


# 결측값 정보를 출력합니다.
df_clean.isnull().sum()


# In[ ]:


# 아래 3개의 변수들의 결측값 정보를 알아보고 싶어서 그 데이터들을 출력합니다.
df_clean[df_clean['Division Name'].isnull()]


# ### 2.2. 결측값 처리

# 이번 실습에서는 `Review Text` 에 있는 데이터만을 머신러닝 입력으로 사용할 것입니다. 그렇기에 `Review Text`의 결측값이 있는 샘플을 삭제합니다.
# 
# 만약 `Review Text` 이외의 데이터를 추가로 사용한다면 추가로 결측값 처리가 수행되어야 할 것입니다.

# In[ ]:


# 결측값이 아닌 부분을 골라내어 df_clasn에 저장합니다.
df_clean = df_clean[~df_clean['Review Text'].isnull()]


# In[ ]:


# 결측값 정보를 출력합니다.
df_clean.isnull().sum()


#  

# ## 3. 데이터 시각화

# 각 변수 분포를 알아보기 위하여 시각화를 수행하겠습니다.
# 
# 일반적으로는 막대그래프를 그리는 방법으로 시각화를 수행하나,
# 
# 문자열로 이루어진 `Title` 데이터와 `Review Text` 데이터는 word cloud란 방식을 사용하여 시각화를 수행하겠습니다.

# ### 3.1. `Title` word cloud

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re


# In[ ]:


# 'Title'의 결측값을 삭제합니다.
df_clean_title = df_clean[~df_clean['Title'].isnull()]

# findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져옵니다.(소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean_title['Title'].str.lower().str.cat(sep=' ')) 


# In[ ]:


# nltk에서 지원하는 'stopwords'를 다운받습니다.
nltk.download('stopwords')


# In[ ]:


# 영어 'stopwords'를 가져옵니다.
en_stops = set(stopwords.words('english'))

# tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장합니다.
filtered_sentence = [token for token in tokens if not token in en_stops]
filtered_sentence


# In[ ]:


# 출력 사이즈를 설정합니다.
plt.rcParams['figure.figsize'] = (16, 16)

# wordcloud를 저장합니다.
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(' '.join(filtered_sentence))

# wordcloud를 출력합니다.
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()


# ### 3.2. `Review Text` word cloud

# In[ ]:


# findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져옵니다.(소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean['Review Text'].str.lower().str.cat(sep=' ')) 

# tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장합니다.
filtered_sentence = [token for token in tokens if not token in en_stops]
filtered_sentence


# In[ ]:


# 출력 사이즈를 설정합니다.
plt.rcParams['figure.figsize'] = (16, 16)

# wordcloud를 저장합니다.
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(filtered_sentence))

# wordcloud를 출력합니다.
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()


#  

# ### 3.3. `Recommended IND` 시각화

# In[ ]:


# 분포를 막대 그래프를 사용하여 출력합니다.
df_clean['Recommended IND'].value_counts().plot(kind='bar')

# 분포를 도수분포표로 확인합니다.
df_clean['Recommended IND'].value_counts()


#  

#  

# ## 4. 데이터 전 처리

# 상품 추천 여부 예측을 수행하기 위해서 주어진 이커머스 데이터에 대해서 분류 모델을 사용할 것입니다.
# 
# 이번 파트에서는 `Review Text` 데이터만을 사용하여 예측을 수행하지만, `age`나 `Title` 데이터 등을 추가하여 입력을 확장할 수 있습니다.
# 
# 분류 모델의 필요한 입력 데이터를 준비 하기위해서 다음과 같은 전처리를 수행하겠습니다.
# 
# 1. `Review Text` 데이터 자연어 전 처리 - Tfidf 활용
# 2. 학습 데이터와 테스트 데이터로 나누기

# ### 4.1. 자연어 전 처리 - Tfidf

# 자연어 전 처리는 자연어 데이터를 머신러닝의 입력값으로 사용할 수 있도록 변환하는 과정을 의미합니다. 이번 실습에서는 sklearn에서 제공하는 `TfidfVectorizer`를 사용하여 문자열 데이터를 수치 자료형 벡터로 변환해 보겠습니다.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer을 불러옵니다. (stop_words 는 영어로 설정)
vectorizer = TfidfVectorizer(stop_words = 'english')

# 소문자화 'Review Text'데이터를 Tfidf로 변환합니다.
X = vectorizer.fit_transform(df_clean['Review Text'].str.lower())


# In[ ]:


# 변환된 X의 크기를 살펴봅니다.
X.shape


# In[ ]:


# 예측해야 할 변수 'Recommended IND' 만을 선택하여 numpy 형태로 y에 저장합니다.
y = df_clean['Recommended IND']
y = y.to_numpy().ravel() # 1 차원 벡터 형태로 출력하기 위해 ravel 사용


# In[ ]:


vectorizer.get_feature_names()


# ### 4.2. 학습, 테스트 데이터 분리

# 머신러닝의 성능을 평가 하기 위해서는 전체 데이터를 학습에 사용하지 않고 학습용 데이터와 테스트용 데이터를 나누어 사용합니다.

# In[ ]:


from sklearn.model_selection import train_test_split

# sklearn에서 제공하는 train_test_split을 사용하여 손 쉽게 분리 할 수 있습니다.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#  

# ## 5. 머신러닝 모델 학습

# 전처리된 데이터를 바탕으로 분류 모델을 학습을 수행하고 학습 결과를 출력 해봅니다.
# 
# 먼저 기본적인 분류 모델인 **의사결정나무(Decision Tree)**를 사용하여 학습을 수행하고, 다양한 모델들을 살펴봅시다.

# ### 5.1. 기본 분류 모델 학습 - 의사결정나무

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# 의사결정나무 DecisionTreeClassifier class를 가져 옵니다.
model = DecisionTreeClassifier()

# fit 함수를 사용하여 데이터를 학습합니다.
model.fit(x_train, y_train)


# In[ ]:


# score 함수를 사용하여 모델의 성능을 출력합니다.
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))


# ### 5.2. 다양한 분류 모델 학습

# 의사결정나무 모델 이외의 다양한 분류 알고리즘을 사용하고 그 성능을 비교하여 봅시다.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


models = []
models.append(('KNN', KNeighborsClassifier()))  # KNN 모델
models.append(('NB-M', MultinomialNB()))  # 멀티노미얼 나이브 베이즈
models.append(('NB-B', BernoulliNB()))  # 베르누이 나이브 베이즈 모델
models.append(('RF', RandomForestClassifier()))  # 랜덤포레스트 모델
models.append(('SVM', SVC(gamma='auto')))  # SVM 모델
models.append(('XGB', XGBClassifier()))  # XGB 모델

for name, model in models:
    model.fit(x_train, y_train)
    msg = "%s - train_score : %f, test score : %f" % (name, model.score(x_train, y_train), model.score(x_test, y_test))
    print(msg)


# In[ ]:


# xgb 모델에서 변수 중요도를 출력합니다.
max_num_features = 20
ax = xgb.plot_importance(models[-1][1], height = 1, grid = True, importance_type = 'gain', show_values = False, max_num_features = max_num_features)
ytick = ax.get_yticklabels()
word_importance = []
for i in range(max_num_features):
    word_importance.append(vectorizer.get_feature_names()[int(ytick[i].get_text().split('f')[1])])

ax.set_yticklabels(word_importance)

plt.rcParams['figure.figsize'] = (10, 15)
plt.xlabel('The F-Score for each features')
plt.ylabel('Importances')
plt.show()


# **퀴즈1. 위 학습된 XGBClassifier 모델에서 feature importance가 30번째인 토큰을 문자열 형태로 구하세요.**
# 
# 첫 번째 feature importance에 해당하는 토큰을 문자열 형태로 하면 'cheap' 입니다.
# 

# In[ ]:


# plot_importance() 함수의 인자 값을 조절하면 쉽게 구할 수 있습니다.


# In[ ]:


# str형으로 문자열만 저장합니다. 예시: quiz_1 = 'cheap'
quiz_1 = None


#  

# ## 6. 평가 및 예측

# 학습 과정에서 학습 데이터와 테스트 데이터에 대해서 accuracy 계산하여 평가하였습니다.
# 
# accuracy의 경우 아래 식에서 알 수 있듯이 얼마나 정확히 예측했는가를 정량적으로 나타냅니다.
# 
# $Accuracy = \frac{Number \;of \;correct \;predictions}{Total \; number \;of \;predictions} $
# 
# 현재 데이터는 `추천을 한다(0)` 는 데이터가 `추천을 하지 않는다(1)` 데이터에 비해 월등히 많은 상황입니다.
# 
# 이런 경우, `추천 한다(0)`만을 정확히 예측해도 높은 accuracy 값을 가질 수 있습니다.  
# 
# 그렇기에 이번 실습에서는 또 다른 성능 지표인 recall 값 또한 살펴봐야 합니다.
# 
# recall 방식은 `추천을 하지 않는다(1)` 대비 `추천을 한다(0)`의 비율을 나타내기에 accuracy에서 놓칠 수 있는 결과 해석을 보충합니다.
# 
# 이번 파트에서는 recall 방식을 포함한 또 다른 대표적인 평가 방법에 대해서 알아보고 주어진 데이터에 대해서 예측하는 것을 수행해보겠습니다.

# ### 6.1. Confusion Matrix

# 기존 score에서 볼 수 있었던 결과는 accuracy 기반의 결과였습니다. confusion matrix를 출력하여 각 class 별로 예측한 결과에 대해서 자세히 알아봅시다.

# In[ ]:


from sklearn.metrics import confusion_matrix

# 의사결정나무 모델에 confusion matrix를 사용하기 위하여 테스트 데이터의 예측값을 저장합니다.
model_predition = model.predict(x_test)

# sklearn에서 제공하는 confusion_matrix를 사용합니다.
cm = confusion_matrix(y_test, model_predition)

# 출력 파트 - seaborn의 heatmap을 사용
plt.rcParams['figure.figsize'] = (5, 5)
sns.set(style = 'dark', font_scale = 1.4)
ax = sns.heatmap(cm, annot=True)
plt.xlabel('Real Data')
plt.ylabel('Prediction')
plt.show()
cm


# 위 confusion matrix에서 x 축은 실제 데이터의 label을 의미하고 y 축은 예측한 데이터의 label을 의미합니다.
# 
# - **0,0 의 값:** `추천함(Pass)` 이라고 예측했을 때, 실제 데이터가 `추천함(Pass)`인 경우의 개수
# - **0,1 의 값:** `추천 하지 않음(Fail)` 이라고 예측했을 때, 실제 데이터가 `추천함(Pass)`인 경우의 개수
# - **1,0 의 값:** `추천함(Pass)` 이라고 예측했을 때, 실제 데이터가 `추천 하지 않음(Fail)`인 경우의 개수
# - **1,1 의 값:** `추천 하지 않음(Fail)` 이라고 에측했을 때, 실제 데이터가 `추천 하지 않음(Fail)`인 경우의 개수

# **퀴즈2. XGBClassifier 모델에서 평가용 데이터(`x_test, y_test`)의 confusion matrix를 구하세요.**

# In[ ]:


# XGBClassifier의 x_test에 대한 예측값을 구하고 confusion_matrix() 를 사용하면 confusion matrix를 구할 수 있습니다.


# In[ ]:


# confusion_matrix() 결과값을 저장합니다. 
quiz_2 = None


#  

# ### 6.2. Precision & Recall

# 분류 모델의 또 다른 성능 지표로 Precsion과 Recall를 구하여 봅시다.

# In[ ]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# sklearn에서 제공하는 recall_score, precision_score를 사용하여 recall과 precision 결과물을 출력합니다.
print("Recall score: {}".format(recall_score(y_test, model_predition)))
print("Precision score: {}".format(precision_score(y_test, model_predition)))


# ### 6.3. 테스트 데이터의 예측값 출력

# In[ ]:


# 0번부터 4번까지 5개를 출력해보겠습니다.
for i in range(5): 
    
    # 의사결정나무 모델을 사용하였습니다.
    prediction = model.predict(x_test[i])
    print("{} 번째 테스트 데이터 문장: \n{}".format(i, df_clean['Review Text'][i]))
    print("{} 번째 테스트 데이터의 예측 결과: {}, 실제 데이터: {}\n".format(i, prediction[0], y_test[i]))


# ---

# ## 제출하기
# 
# 퀴즈 1번, 2번까지 수행 후, 아래 코드를 실행하면 `quize_1, quize_2` 변수가 저장된 `submission.pickle` 파일을 제작하여 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


import pickle

d = {'quiz_1': quiz_1, 'quiz_2': quiz_2}

with open('submission.pickle', 'wb') as f:
    pickle.dump(d, f)


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

# ## 챌린지
# 
# 이번 프로젝트에서 사용한 모델은 학습용 데이터에 대한 **accuracy**는 좋은 성능을 보여주었지만, 평가용 데이터에서 **recall, precision** 지표는 좋은 성능을 보여주지 못 하였습니다. 다양한 방식을 사용하여 평가용 데이터에서의 **recall, precision**을 높일 수 있는 방법을 찾아봅시다.

# ---

# <span style="color:rgb(120, 120, 120)">본 학습 자료를 포함한 사이트 내 모든 자료의 저작권은 엘리스에 있으며 외부로의 무단 복제, 배포 및 전송을 불허합니다.
# 
# Copyright @ elice all rights reserved</span>
