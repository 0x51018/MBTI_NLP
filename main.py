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
from tensorflow.keras.layers import Embedding, Dense, LSTM, GRU, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
    
import tkinter as tk
import sys
from tkinter import ttk
from PIL import ImageTk, Image
from ctypes import windll
import time
windll.shcore.SetProcessDpiAwareness(1) #tkinter 해상도 개선

questions, answers = [], []
default_font = 'KoPubWorld돋움체 Medium'

result_chk = [["E", 'I'], ["N", "S"], ["T", "F"], ["P", "J"]] 
result_ratio = []
result=""
bgcolor, fgcolor = "#ffffff", "#000000"


ko_model = gensim.models.Word2Vec.load("X:\\Anaconda\\envs\\main\\Lib\\site-packages\\gensim\\models\\ko\\ko.bin") #Word2Vec 파일 설치 경로

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len): cnt += 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
 

S_answer0, A_answer0, S_label0, A_label0 = [], [], [], []
S_ACC, A_ACC = [[], [], [], []], [[], [], [], []]
S_QUE, A_QUE = [["S", "E/I"], ["S", "N/S"],["S", "T/F"],["S", "P/J"]], [["A", "E/I"], ["A", "N/S"],["A", "T/F"],["A", "P/J"]]

data_list = list(csv.reader(open('data_fin.csv', 'r', encoding='utf-8')))

for line in data_list:
    if(line[7] == 'S'):
        S_answer0.append(line[8:29])
        S_label0.append([1 if (line[3] == 'E') else 0, 1 if (line[4] == 'N') else 0, 1 if (line[5] == 'T')else 0, 1 if (line[6] == 'P')else 0])
    if(line[7] == 'A'):
        A_answer0.append(line[29:50])
        A_label0.append([1 if (line[3] == 'E') else 0, 1 if (line[4] == 'N') else 0, 1 if (line[5] == 'T')else 0, 1 if (line[6] == 'P')else 0])

S_answer, A_answer = np.array(list(map(list, zip(*S_answer0)))), np.array(list(map(list, zip(*A_answer0))))
S_label, A_label = np.array(list(map(list, zip(*S_label0)))), np.array(list(map(list, zip(*A_label0))))

mecab = Mecab(dicpath=r"C:\\mecab\\mecab-ko-dic") # mecab 설치 경로

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title('MBTI 검사 챗봇')
        self._frame = None
        self.geometry("1600x900+10+10")
        self.resizable(0,0)
        self.configure(bg=bgcolor)
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None: self._frame.destroy()
        self._frame = new_frame
        self._frame.pack(side="top", fill="both")

class StartPage(tk.Frame):
    def __init__(self, master):
        global  r
        r = 0
        self.a = master
        self.reset()

        while(r == 0): r = SelectQuestion()
        for _ in Q: questions.append(_)

        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, bg=bgcolor)
        tk.Label(self, text="대화형 MBTI 검사", font=(default_font, 32, "bold"), fg=fgcolor, bg=bgcolor).pack(side="top", fill="x", pady=(150,0))
        tk.Label(self, text="새로운 형태의 MBTI 검사 시스템에 오신걸 환영합니다!\n여러분은 15개의 질문에만 답하여 MBTI를 검사할 수 있습니다.\n각각의 질문에 평소 생각하는 것처럼 편하게 답변해보세요!\n", font=(default_font, 15), fg="#888888", bg=bgcolor).pack(side="top", fill="x", pady=(50,300-50))
        tk.Button(self, text="검사 시작", font=(default_font, 13), relief="flat", command=lambda:self.switch_frame(), width=22, bg=fgcolor, fg=bgcolor).pack(side="bottom", pady=(10, 0))

    def switch_frame(self): 
        self.a.switch_frame(ProgressPage)
    
    def reset(self):
        global answers, questions, result_ratio, result
        answers = []
        questions = []
        result_ratio = []
        result=""

class ProgressPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, bg=bgcolor)
        self.count = 0
        self.a = master
        self.btn_able = 0
        global answers
        answers = []

        s = ttk.Style()
        s.theme_use('clam')
        s.configure("red.Horizontal.TProgressbar", background='darkblue', troughcolor='white', darkcolor='darkblue', lightcolor='darkblue')

        self.k = tk.Frame(self, bg=bgcolor)
        self.k2 = tk.Frame(self, bg=bgcolor, borderwidth=5, relief="groove")
        self.p = ttk.Progressbar(self, style="red.Horizontal.TProgressbar", length=1100, maximum=len(questions), value = 0)
        self.c = tk.Label(self.k, text=questions[self.count]+"\n"*(3-questions[self.count].count('\n')), font=(default_font, 18, "bold"), bg=bgcolor, fg=fgcolor)
        self.e = tk.Entry(self.k2, width=30, font=(default_font, 18), relief="flat")
        self.e.bind("<Return>", self.next_btn)
        
        self.bt = tk.Button(self.k2, text="→", font=(default_font, 9), width=2, height=1, command=lambda:self.next(), bg='navy', fg=bgcolor)
        self.bt.pack(side="right", padx=(5,5))
        self.p.pack(side="top", pady=(20, 60))
        self.k.pack(side="top", fill='both', pady=(0,400))
        self.c.pack(side="top", fill="x")
        self.k2.pack(side="bottom", pady=(0, 80))
        self.e.pack(side="left")

    def next_btn(self, event): self.next()

    def next(self): 
        self.fuck = self.e.get()
        if self.fuck != '' and self.btn_able == 0:
            self.btn_able = 1
            self.p.step(1)
            if self.count >= len(questions)-1:
                answers.append(self.fuck)
                self.switch_frame()
            else:
                self.c.config(text=questions[self.count+1]+"\n"*(3-questions[self.count+1].count('\n')))
                answers.append(self.fuck)
                self.e.delete(0, len(self.e.get()))
            self.btn_able = 0
            self.count += 1
    def switch_frame(self):
        self.a.switch_frame(Result0Page)

class Result0Page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, bg=bgcolor)
        tk.Label(self, text="테스트가 종료되었습니다. \n답변을 기반으로 결과를 생성하고 있습니다.\n\n잠시만 기다려주세요.", font=(default_font, 18, "bold"), bg=bgcolor, fg=fgcolor).pack(side="top", fill="x", pady=(100, 5))
        self.a = master
        print(answers)

        s = ttk.Style()
        s.theme_use('clam')
        s.configure("red.Horizontal.TProgressbar", background='darkblue', troughcolor='white', darkcolor='darkblue', lightcolor='darkblue')

        self.p = ttk.Progressbar(self, style="red.Horizontal.TProgressbar", length=1100, maximum=len(answers), value = 0)
        self.p.pack(side="top", pady=(450, 0))
        
        self.after(1000, self.processing)
        
    def processing(self):
        for i in range(len(answers)):
            try: evaluate(answers[i], Q_coordinate[i])
            except: print("error : len(answers) =", len(answers), ", len(Q_coordinate) =", len(Q_coordinate), ", i=", i)
            
            self.p.configure(value = i+1)
            self.update()
        predictMBTI() # 결과출력
        tk.Button(self, text="결과 보기", font=(default_font, 12), bg=fgcolor, fg=bgcolor,
                  command=lambda: self.a.switch_frame(ResultPage)).pack(side="top", pady=(50, 0))

       
class ResultPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Frame.configure(self, bg=bgcolor, width=1200, height=675)

        result = ''
        for x in range(4):
            if result_ratio[x*2] > 50: result = result+result_chk[x][0]
            else: result = result+result_chk[x][1]
        
        tk.Label(self, text="당신의 MBTI는 "+result+"입니다!", font=(default_font, 18, "bold"), bg=bgcolor, fg=fgcolor).pack(side="top", fill="x", pady=25)
        
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("blue.Horizontal.TProgressbar", background='darkblue', troughcolor='white', darkcolor='darkblue', lightcolor='darkblue')
        
        self.fucku = []
        for x in range(4):
            a, b = result_chk[x]
            self.fucku.append(tk.Label(self, text=a, font=(default_font, 10), bg=bgcolor, fg=fgcolor, anchor="e"))
            self.fucku.append(tk.Label(self, text=b, font=(default_font, 10), bg=bgcolor, fg=fgcolor, anchor="e"))

        down = 45

        canvas = tk.Canvas(self, width=1200, height=450, bg="white", bd=0, highlightthickness=0)
        canvas.create_rectangle(250, 35+down, 950, 52+down, width=1, outline="#9E9A91")
        canvas.create_rectangle(250, 135+down, 950, 152+down, width=1, outline="#9E9A91")
        canvas.create_rectangle(250, 235+down, 950, 252+down, width=1, outline="#9E9A91")
        canvas.create_rectangle(250, 335+down, 950, 352+down, width=1, outline="#9E9A91")

        canvas.create_text(260, 20+down, text=result_chk[0][0], fill="#00008b", font=(default_font, 12), width=1)
        canvas.create_text(940, 20+down, text=result_chk[0][1], fill="#00008b", font=(default_font, 12), width=1)
        canvas.create_text(260, 120+down, text=result_chk[1][0], fill="#00008b", font=(default_font, 12), width=1)
        canvas.create_text(940, 120+down, text=result_chk[1][1], fill="#00008b", font=(default_font, 12), width=1)
        canvas.create_text(260, 220+down, text=result_chk[2][0], fill="#00008b", font=(default_font, 12), width=1)
        canvas.create_text(940, 220+down, text=result_chk[2][1], fill="#00008b", font=(default_font, 12), width=1)
        canvas.create_text(260, 320+down, text=result_chk[3][0], fill="#00008b", font=(default_font, 12), width=1)
        canvas.create_text(940, 320+down, text=result_chk[3][1], fill="#00008b", font=(default_font, 12), width=1)

        canvas.create_text(270, 72+down, text=str(round(result_ratio[0],1))+"%", fill="#00008b", font=(default_font, 10), width=60)
        canvas.create_text(930, 72+down, text=str(round(result_ratio[1],1))+"%", fill="#00008b", font=(default_font, 10), width=60)
        canvas.create_text(270, 172+down, text=str(round(result_ratio[2],1))+"%", fill="#00008b", font=(default_font, 10), width=60)
        canvas.create_text(930, 172+down, text=str(round(result_ratio[3],1))+"%", fill="#00008b", font=(default_font, 10), width=60)
        canvas.create_text(270, 272+down, text=str(round(result_ratio[4],1))+"%", fill="#00008b", font=(default_font, 10), width=60)
        canvas.create_text(930, 272+down, text=str(round(result_ratio[5],1))+"%", fill="#00008b", font=(default_font, 10), width=60)
        canvas.create_text(270, 372+down, text=str(round(result_ratio[6],1))+"%", fill="#00008b", font=(default_font, 10), width=60)
        canvas.create_text(930, 372+down, text=str(round(result_ratio[7],1))+"%", fill="#00008b", font=(default_font, 10), width=60)

        p1, p2, p3, p4 = (947-253+1)/100*result_ratio[0], (947-253+1)/100*result_ratio[2], (947-253+1)/100*result_ratio[4], (947-253+1)/100*result_ratio[6] 
        if result_ratio[0] > 50: canvas.create_rectangle(253, 38+down, 252+p1, 49+down, fill="#00008b")
        else: canvas.create_rectangle(253+p1, 38+down, 947, 49+down, fill="#00008b")
        if result_ratio[2] > 50: canvas.create_rectangle(253, 138+down, 252+p2, 149+down, fill="#00008b")
        else: canvas.create_rectangle(253+p2, 138+down, 947, 149+down, fill="#00008b")
        if result_ratio[4] > 50: canvas.create_rectangle(253, 238+down, 252+p3, 249+down, fill="#00008b")
        else: canvas.create_rectangle(253+p3, 238+down, 947, 249+down, fill="#00008b")
        if result_ratio[6] > 50: canvas.create_rectangle(253, 338+down, 252+p4, 349+down, fill="#00008b")
        else: canvas.create_rectangle(253+p4, 338+down, 947, 349+down, fill="#00008b")


        canvas.pack()

        tk.Button(self, text="시작 화면으로 돌아가기", font=(default_font, 12), bg=fgcolor, fg=bgcolor,
                  command=lambda: master.switch_frame(StartPage)).pack(side="bottom", pady=(250,0))


ff = open('acc_fin.csv', 'r', encoding='utf-8')
data2 = csv.reader(ff)
data2 = list(data2)

S_elite=[[], [], [], []]
A_elite=[[], [], [], []]
Q_ACC=[[], [], [], []]

j = 0
SS = [0 for i in range(21)]
AA = [0 for i in range(21)]

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

#print(*S_elite, sep = '\n')
#print(*A_elite, sep = '\n')

#print(sum(SS))
#print(sum(AA))

Check = SS+AA # 유용한 질문인지 저장
#print(len(Check))
AllQuestions = [S_elite[i] + A_elite[i] for i in range(4)] # 레이블별 질문 저장, S : 0~20, A : 21~41
#print(AllQuestions)
eliteQ = list(set().union(*AllQuestions))
#print(eliteQ)

AQ, Q = [], [] # 전체 질문 문자열 저장, 선별된 질문 문자열 저장
Selected = []
Q_coordinate = [] # 선별된 질문의 번호를 저장함
MBTI = [[], [], [], []] # MBTI 레이블별 선별된 질문의 번호. 균형을 맞추기 위해 사용

#설문 데이터 불러오기
f = open('data_fin.csv', 'r', encoding='utf-8')
data = csv.reader(f)
data_list = list(data)

for i in range(0, 42):
    k = data_list[0][i+8].find(" ") + 1
    AQ.append(data_list[0][i+8][k::])

def SelectQuestion(): # 랜덤하게 질문을 고르는 함수
    Selected = []
    global Q, Q_coordinate, MBTI
    Q = []
    Q_coordinate = []
    MBTI = [[], [], [], []] 
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

Score, Max_Score = [0, 0, 0, 0], [0, 0, 0, 0]
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
        
def predictMBTI():
    Path("txtfile").mkdir(parents=True, exist_ok=True)
    a = 1
    while(os.path.isfile("txtfile\\data%d.txt"%a)): a += 1
    f = open("txtfile\\data%d.txt"%a, 'w')
    for i in Q: f.write(i+"\n")
    for i in answers: f.write(i+"\n")
    f.close()
    global result
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

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()