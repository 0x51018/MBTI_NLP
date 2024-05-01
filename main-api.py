#Library imports
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path
import pickle
import random as rd
import gensim
from konlpy.tag import Mecab
from keras import models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import sys
from ctypes import windll
import time

from flask import Flask, render_template, request, jsonify, Response, session
from flask_session import Session

app = Flask(__name__) #Flask 앱 생성

#전역 데이터 초기화 작업들
ko_model = gensim.models.Word2Vec.load("X:\\Anaconda\\envs\\main\\Lib\\site-packages\\gensim\\models\\ko\\ko.bin") #Word2Vec 파일 설치 경로
mecab = Mecab(dicpath=r"C:\\mecab\\mecab-ko-dic") # mecab 설치 경로
data2 = list(csv.reader(open('acc_fin.csv', 'r', encoding='utf-8')))

S_elite, A_elite = [[], [], [], []],[[], [], [], []]
Q_ACC=[[], [], [], []]

j = 0
SS, AA = [0 for i in range(21)], [0 for i in range(21)]
for line in data2:
    if(line[0] == 'S'):
        for i in range(2, 23):
            Q_ACC[j].append(float(line[i]))
            if(float(line[i]) > 0.6):
                S_elite[j].append(i-2)
                SS[i-2] = 1
        j+=1
    if(line[0] == 'A'):
        for i in range(2, 23):
            Q_ACC[j%4].append(float(line[i]))
            if(float(line[i]) > 0.6):
                A_elite[j%4].append(i+19)
                AA[i-2] = 1
        j+=1

Check = SS+AA
AllQuestions = [S_elite[i] + A_elite[i] for i in range(4)] # 레이블별 질문 저장, S : 0~20, A : 21~41
eliteQ = list(set().union(*AllQuestions))

AQ = [] # 전체 질문 문자열 저장

data_list = list(csv.reader(open('data_fin.csv', 'r', encoding='utf-8'))) #설문 데이터 불러오기
for i in range(0, 42):
    k = data_list[0][i+8].find(" ") + 1
    AQ.append(data_list[0][i+8][k::])

#클라이언트(세선)마다 따로 저장해야 할 것들
Q, Q_coordinate = [], [] # 각각 선별된 질문의 문자열, 번호를 저장함
Score, Max_Score = [0, 0, 0, 0], [0, 0, 0, 0] #점수, 최대 점수
answers, questions, result_ratio = [], [], []
MBTI = [[], [], [], []] # MBTI 레이블별 선별된 질문의 번호. 균형을 맞추기 위해 사용

def SelectQuestion(): # 랜덤하게 질문을 고르는 함수
    Selected = []
    global Q, Q_coordinate, MBTI
    Q, Q_coordinate, MBTI = [], [], [[], [], [], []] 
    for i in range(15):
        temp = 0
        while(temp == 0):
            k = rd.choice(eliteQ)
            temp = Check[k] if k not in Selected else 0
        Selected.append(k)
        for j in range(4): 
            if k in AllQuestions[j]: MBTI[j].append(k)
        Q.append(AQ[k])
        Q_coordinate.append(k)
    if(min([len(MBTI[i]) for i in range(4)]) < 4): return 0
    else: return 1
def evaluate(inputstr, i): # i가 질문의 번호임!
    stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '.' , ',', '?', '!']
    inputstr = inputstr.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    inputstr = inputstr.replace('^ +', "")

    temp = mecab.morphs(inputstr)
    sentence = [word for word in temp if not word in stopwords]
    sentence_0 = []
    sentence_0.append(sentence)
    print(sentence_0)

    tokenizer = Tokenizer()
    if i>20:
        with open("models\A_dict%d.pickle"%(i-21), "rb") as fr: tokenizer = pickle.load(fr)
    else:
        with open("models\S_dict%d.pickle"%i, "rb") as fr: tokenizer = pickle.load(fr)

    sentence_1 = tokenizer.texts_to_sequences(sentence_0)
    max_len = 67 if i>20 else 57
    sentence_2 = pad_sequences(sentence_1, maxlen = max_len)
    print(sentence_2)

    for j in range(4):
        if i in MBTI[j]:
            ACC = 0
            if i>20: model = models.load_model("models\A_Q%d_LABEL%d.h5"%(i-21, j))
            else: model = models.load_model("models\S_Q%d_LABEL%d.h5"%(i, j))
                
            k = model.predict(sentence_2)
            k = k[0][0]
            ACC = Q_ACC[j][i] - 0.5
            Max_Score[j] += ACC
            Score[j] += (ACC*k)
r = 0
while(r == 0): r = SelectQuestion()
for _ in Q: questions.append(_)

# 기능 A: 질문 15개를 사용자에게 보내기
@app.route('/get_questions', methods=['GET'])
def get_questions():
    print("질문을 올바르게 전송하였음.")
    return jsonify({"questions": Q})

# 기능 B: 답변 15개를 사용자로부터 받기
@app.route('/send_answers', methods=['POST'])
def send_answers():
    print("답변 수신 대기 중")
    data = request.get_json()
    
    if 'answers' in data and len(list((data)['answers'])) == 15:
        data = list((request.get_json())['answers'])  # JSON 데이터를 받음
        print("data:",data, "  data type:", type(data))
        for _ in range(15): answers.append(data[_])
        print("answers:", answers)
        print("15개의 답변을 저장했습니다.")
        return jsonify({"message": "15개의 답변을 저장했습니다."})
    else:
        if 'answers' not in data:
            print("올바른 데이터를 수신하지 못했습니다: 수신된 데이터가 answers가 아닙니다.")
            return jsonify({"error": "올바른 데이터를 전송하지 않았습니다. 전송한 데이터가 answer이 아닙니다."}), 400
        if 'answers' in data and len(list((data)['answers'])) != 15:
            print("올바른 데이터를 수신하지 못했습니다. 수신된 answers가", len(list((data)['answers'])), "개입니다. (required : 15)")
            return jsonify({"error": "올바른 데이터를 전송하지 않았습니다. 유효하지 않은 개수의 answers를 전송하였습니다."}), 400
#
def send_progress():
    progress = 0
    while progress <= 15:
        try: evaluate(answers[progress], Q_coordinate[progress])
        except: print("error : len(answers) =", len(answers), ", len(Q_coordinate) =", len(Q_coordinate), ", progress=", progress)
        progress += 1
        time.sleep(0.5)
        message = f"data: {int(progress/15 * 100)}%\n\n"
        yield message

# 기능 C: 답변을 통해 결과를 계산하고, 진행 상황을 보내기
@app.route('/get_progress', methods=['GET'])    
def get_progress():
    return Response(send_progress(), content_type='text/event-stream')

# 기능 D: 산출된 결과(mbti)를 보내기
@app.route('/get_result', methods=['GET'])
def get_result():
    Path("txtfile").mkdir(parents=True, exist_ok=True)
    a = 1
    while(os.path.isfile("txtfile\\data%d.txt"%a)): a += 1
    f = open("txtfile\\data%d.txt"%a, 'w')
    for i in Q: f.write(i+"\n")
    for i in answers: f.write(i+"\n")
    f.close()
    result = ""
    p="ENTP" # 점수가 0.5 이상일 때 (점수는 100점 만점으로 환산했음, 따라서 50점 이상일 때)
    q="ISFJ" # 점수가 0.5 미만일 때
    for i in range(4):
        f_score  = round((Score[i] / Max_Score[i]), 2) * 100 # E, N, T, P 점수(100점 만점)
        if i == 0:  f_score += 6 #데이터 편향 해결
        if i == 3: f_score -= 3 #데이터 편향 해결

        if(f_score >= 50): result = result + p[i]
        else: result = result + q[i]
        # 위 if문은 예측된 MBTI를 문자열로 변환하는 부분
        result_ratio.extend([f_score, 100-f_score])
    return jsonify({"result": result})

# 기능 E: 사용 가능한 상태인지 확인
@app.route('/check_status', methods=['GET'])
def check_status():
    # 기능 B, C, D의 상태를 확인하고, 상태에 따라 응답을 생성하세요.
    return jsonify({"status": "사용 가능"})

# 기본: 시작 화면 웹페이지 출력 
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)