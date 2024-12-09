import os
import sys
import random
import time
import argparse
import numpy as np

from model.xgboost_WbyT import GradientBoostingMultiClassifier
from model.utils import util
from sklearn.metrics import accuracy_score

folder = './results/'

def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    setter = util(seed=args.seed, n_clients=args.n_clients)

    client_set, test_set = setter.get_datadst_setting(filename=args.filename,
                                                      args=args)

    model = GradientBoostingMultiClassifier(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                            outdim=setter.outdim, lamb=args.lamb,
                                            hes=args.hes)

    model.fit(client_set, test_set, args)
    testX, _, testY = test_set
    y_hat = model.predict(testX)
    acc = accuracy_score(testY, y_hat)
    print("ACC:", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated_XGBoost_WbyT')
    parser.add_argument('--seed', default=0, type=int, help='랜덤 시드 설정')
    parser.add_argument('--filename', default="drybean", type=str, help='데이터 파일명')

    parser.add_argument('--joint', default=True, type=bool, help="False: 중앙집중식 학습, True: 연합 학습")
    parser.add_argument('--n_clients', default=10, type=int, help='클라이언트 수')
    parser.add_argument('--env', default='label_nonIID', type=str, help='클라이언트 데이터 환경 설정', choices=['IID', 'label_nonIID'])
    parser.add_argument('--alpha', default=1, type=int, help='Non-IID 데이터 환경에서 불균형 조절 매개변수')
    
    parser.add_argument('--trees', default=100, type=int, help='의사 결정 트리의 수')
    parser.add_argument('--depth', default=10, type=int, help='트리의 최대 깊이')
    parser.add_argument('--count', default=2, type=int, help='이진 분할에 필요한 최소 데이터 수')
    parser.add_argument('--fraction', default=1.0, type=float, help='트리 당 샘플링 사이즈 비율')
    parser.add_argument('--lr', default=0.3, type=float, help='학습률')    
    parser.add_argument('--hes', default=True, type=bool, help="False: 미구현, True: 대각 행렬을 가정하여 헤시안 행렬 계산")
    parser.add_argument('--lamb', default=1.0, type=float, help="규제 매개변수")
    
    parser.add_argument('--method', default='GinM', type=str, help="트리 학습 방법", choices=['GinO', 'GinM'])
    parser.add_argument('--levelUp', default=1, type=int, help='클라이언트 당 추가 트리 높이 ')
    parser.add_argument('--init_depth', default=1, type=int, help='첫 클라이언트에서 생성할 트리의 높이')
    args = parser.parse_args()

    # results 폴더 내 모든 파일을 삭제
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    # 현재 실험의 설정 내용을 텍스트 파일에 쓰기 및 저장
    with open(f"{folder}arguments.txt", 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    run(args)

    pass
