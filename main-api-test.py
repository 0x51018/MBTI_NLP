#Library imports
import csv
import os
import base64
from pathlib import Path
import pickle
import random as rd
import gensim
from konlpy.tag import Mecab
from keras import models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from flask import Flask, render_template, request, stream_with_context, jsonify, Response, session, redirect, url_for
from flask_session import Session

app = Flask(__name__) #Flask 앱 생성
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'mbtinlpaiproject'

app.config['PERMANENT_SESSION_LIFETIME'] = 3600
Session(app)


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
AllQuestions, AQ = [S_elite[i] + A_elite[i] for i in range(4)], [] # 레이블별 질문 저장(S : 0~20, A : 21~41), 전체 질문 문자열 저장
eliteQ = list(set().union(*AllQuestions))
data_list = list(csv.reader(open('data_fin.csv', 'r', encoding='utf-8'))) #설문 데이터 불러오기
for i in range(0, 42):
    k = data_list[0][i+8].find(" ") + 1
    AQ.append(data_list[0][i+8][k::])


def SelectQuestion(): # 랜덤하게 질문을 고르는 함수
    Selected = []
    session['Q'], session['Q_coordinate'], session['MBTI'] = [], [], [[], [], [], []] 
    for i in range(15):
        temp = 0
        while(temp == 0):
            k = rd.choice(eliteQ)
            temp = Check[k] if k not in Selected else 0
        Selected.append(k)
        for j in range(4): 
            if k in AllQuestions[j]: session['MBTI'][j].append(k)
        session['Q'].append(AQ[k])
        session['Q_coordinate'].append(k)
    if(min([len(session['MBTI'][i]) for i in range(4)]) < 4): return 0
    else: return 1
def evaluate(mbti, inputstr, i, score, max_score): # i가 질문의 번호임!
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
    #print(sentence_2)

    for j in range(4):
        if i in mbti[j]:
            ACC = 0
            if i>20: model = models.load_model("models\A_Q%d_LABEL%d.h5"%(i-21, j))
            else: model = models.load_model("models\S_Q%d_LABEL%d.h5"%(i, j))
                
            k = model.predict(sentence_2)
            k = k[0][0]
            ACC = Q_ACC[j][i] - 0.5
            max_score[j] += ACC
            score[j] += k*ACC
    #print("\\\\\\\\\\", score, max_score, "//////////")
    return [score, max_score]

# 기능 A: 질문 15개를 사용자에게 보내기
@app.route('/get_questions', methods=['GET'])
def get_questions():
    print("질문을 올바르게 전송하였음.")
    return jsonify({"questions": session['Q']})

# 기능 B: 답변 15개를 사용자로부터 받기
@app.route('/send_answers', methods=['POST'])
def send_answers():
    print("답변 수신 대기 중")
    data = request.get_json()
    
    if 'answers' in data and len(list((data)['answers'])) == 15:
        data = list((request.get_json())['answers'])  # JSON 데이터를 받음
        print("data:",data, "  data type:", type(data))
        while len(session['answers']): session['answers'].pop()
        for _ in range(15): session['answers'].append(data[_])
        print("answers:", session['answers'])
        print("15개의 답변을 저장했습니다.")
        return redirect(url_for('loading'))
    else:
        if 'answers' not in data:
            print("올바른 데이터를 수신하지 못했습니다: 수신된 데이터가 answers가 아닙니다.")
            return jsonify({"error": "올바른 데이터를 전송하지 않았습니다. 전송한 데이터가 answer이 아닙니다."}), 400
        if 'answers' in data and len(list((data)['answers'])) != 15:
            print("올바른 데이터를 수신하지 못했습니다. 수신된 answers가", len(list((data)['answers'])), "개입니다. (required : 15)")
            return jsonify({"error": "올바른 데이터를 전송하지 않았습니다. 유효하지 않은 개수의 answers를 전송하였습니다."}), 400

# 기능 C: 답변을 통해 결과를 계산하고, 진행 상황을 보내기
@app.route('/get_progress', methods=['GET'])
def get_progress():
    def send_progress(mbti, score, max_score, answers, Q_coordinate, save1, save2):
        progress = 0

        while progress < 15:
            try:
                score, max_score = list(evaluate(mbti, answers[progress], Q_coordinate[progress], score, max_score))[:]
                print("succeed. progress : "+str(progress+1)+'/15')
            except Exception as ex:
                 print(ex)
                 print("answers :", answers)
                 print("Q_coordinate :", Q_coordinate)
                #print(answers, Q_coordinate, progress)
                #print("error : len(answers) =", len(answers), ", len(Q_coordinate) =", len(Q_coordinate), ", progress=", progress)

            progress += 1
            yield f"data:{{\"progress\": {progress}, \"total\": {15}}}\n\n"
            time.sleep(0.5)
        
        yield f"data:{{\"status\": \"finished\", \"score\": {score}, \"maxScore\": {max_score}}}\n\n"
        return [score, max_score]

    return Response(stream_with_context(send_progress(session['MBTI'][:], session['Score'][:], session['Max_Score'][:], session['answers'][:], session['Q_coordinate'][:], session['Score'], session['Max_Score'])), mimetype='text/event-stream')

# 기능 D: 산출된 결과(mbti)를 보내기
@app.route('/get_result', methods=['GET'])
def get_result():
    import json
    score = request.args.get('score')
    max_score = request.args.get('maxScore')
    
    # 문자열을 JSON으로 파싱
    session['Score'] = list(map(float, str(json.loads(score)).split(",")))
    session['Max_Score'] = list(map(float, str(json.loads(max_score)).split(",")))
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", session['Score'], session['Max_Score'], ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #print("ttttttttt", session['Score'], session['Max_Score'], "//////////////////")
    Path("txtfile").mkdir(parents=True, exist_ok=True)
    a = 1
    while(os.path.isfile("txtfile\\data%d.txt"%a)): a += 1
    f = open("txtfile\\data%d.txt"%a, 'w')
    for i in session['Q']: f.write(i+"\n")
    for i in session['answers']: f.write(i+"\n")
    f.close()
    result = ""
    p="ENTP" # 점수가 0.5 이상일 때 (점수는 100점 만점으로 환산했음, 따라서 50점 이상일 때)
    q="ISFJ" # 점수가 0.5 미만일 때
    for i in range(4):
        try: f_score  = round((float(session['Score'][i]) / float(session['Max_Score'][i])), 2) * 100 # E, N, T, P 점수(100점 만점)
        except: print("\\\\\\\\\\\\\\\\\\", session['Score'], session['Max_Score'], "//////////////////")
        if i == 0:  f_score += 6 #데이터 편향 해결
        if i == 3: f_score -= 3 #데이터 편향 해결

        if(f_score >= 50): result = result + p[i]
        else: result = result + q[i]
        # 위 if문은 예측된 MBTI를 문자열로 변환하는 부분
        session['result_ratio'].extend([f_score, 100-f_score])
    return jsonify({"result": result, "result_ratio":session['result_ratio']})

# 기본: 시작 화면 웹페이지 출력 
@app.route('/')
def index():
    session['Q'], session['Q_coordinate'] = [], [] # 각각 선별된 질문의 문자열, 번호를 저장함
    session['Score'], session['Max_Score'] = [0, 0, 0, 0], [0, 0, 0, 0] #점수, 최대 점수
    session['answers'], session['questions'], session['result_ratio'] = [], [], []
    session['MBTI'] = [[], [], [], []] # MBTI 레이블별 선별된 질문의 번호. 균형을 맞추기 위해 사용

    r = 0
    while(r == 0): r = SelectQuestion()
    for _ in session['Q']: session['questions'].append(_)
    return render_template('index.html')

@app.route('/questions')
def questions():
    #클라이언트(세선)마다 따로 저장해야 할 것들
    session['Q'], session['Q_coordinate'] = [], [] # 각각 선별된 질문의 문자열, 번호를 저장함
    session['Score'], session['Max_Score'] = [0, 0, 0, 0], [0, 0, 0, 0] #점수, 최대 점수
    session['answers'], session['questions'], session['result_ratio'] = [], [], []
    session['MBTI'] = [[], [], [], []] # MBTI 레이블별 선별된 질문의 번호. 균형을 맞추기 위해 사용

    r = 0
    while(r == 0): r = SelectQuestion()
    for _ in session['Q']: session['questions'].append(_)
    return render_template('questions.html')

@app.route('/loading', methods=['GET'])
def loading():
    return render_template('loading.html')

@app.route('/result_page', methods=['GET'])
def result_page():
    score_encoded = request.args.get('score', '')
    max_score_encoded = request.args.get('maxScore', '')
    
    score = base64.b64decode(score_encoded).decode('utf-8')
    max_score = base64.b64decode(max_score_encoded).decode('utf-8')
    return render_template('result_page.html', SSScore=score, SSMax_Score=max_score)

if __name__ == '__main__':
    app.run(debug=True, port=5001)