# musical_for_beginners

Fine Tuning을 통해 Interpark 뮤지컬 리뷰들에 대해서 다음의 8가지 카테고리에 대해 각각 내용에 선호정보가 존재하는지 binary classify하는 Multi-Label Text Classification을 진행한다.

category list: { 'funny': '유쾌','touching': '감동','story': '스토리','immersion': '몰입도','stage': '무대','song': '노래','dance': '춤', acting': '연기'}

소스 설명: 

"./mfb_annotation/mfb_annotate.py" : 레이블링 gui 환경


"./model/config.py" : 프로젝트 관련 몇가지 상수 정의
"./model/kcbert_finetune.py": KcBERT를 활용한(with huggingface) fine tuning code 정의
"./model/kobert_finetune.py": KoBERT를 활용한(without huggingface) fine tuning code 정의
"./model/metric_utils" : category별 metric을 각각 도출하기 위한 구현부
"./model/mfb_classifier.py": 완성한 모델을 활용하기 위한 클래스 정의
"./model/mfb_fine_tune.py" : 각 fine tuning code와 연결 정의
"./model/pretrained_loader.py": pre-trained model 로드


"./easy_dat_aug.py" : train-set 증강을 위해 활용
"./mfb_dataset.py": mongodb에서 데이터를 로드하여 데이터셋 형성
"./model_io_translation.py" : model output과 db record간의 translation 정의
(ignored)./mongo_access.py" : db 접속 코드(보안상 생략)


How to run:
  학습이 완료된 pytorch model file의 path를 ./model/config.py상 작성,
  ./model/mfb_classifier의 MusicalForBeginners 코드 참고
  
How to train:
  ./model/config.py상에 fine tuning시 작용할 hyper parameter 지정.
  ./model/mfb_fine_tune.py의 main 부분 참고
