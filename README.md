Effective Federated XGBoost Learning Method for Multi-Class Classification in Non-IID Environments
이 레포지토리는 Non-IID 데이터 분포에서 다중 클래스 분류에 효과적인 연합 XGBoost의 구현 코드를 저장합니다. 구현된 XGBoost는 기존의 스칼라 가중치로 설계된 트리가 아닌 벡터 형태의 가중치를 저장하는 XGBoost[1] 입니다.

[1] N. Ponomareva et al., “Compact Multi-Class Boosted Trees,” in international Conference on Big Data, 2017. paper: https://arxiv.org/abs/1710.11547.

소개
구현된 연합 XGBoost의 접근 방법은 XGBoost의 트리 성장 단계와 트리의 전역 가중치 조정를 저장하는 단계로 나누어져 있습니다. 각 단계와 단계 별로 파생되는 전략을 아래와 같이 요약합니다.

트리 성장 단계: 로컬 데이터를 기반으로 트리의 층을 성장시키는 단계.

GinO(Grow in One client): 한 클라이언트에서 트리의 층을 완전히 성장시켜 구조를 생성하는 전략.
GinM(Grow in Multi clients): 여러 클라이언트를 순회하면서 트리의 층을 성장시켜 구조를 생성하는 전략.
전역 가중치 조정 단계: 트리 성장 단계에서 생성된 트리에 여러 클라이언트의 로컬 그레디언트들을 집계하여 전역 가중치를 저장하는 단계.

WbyT(Weight update by Trees): 트리가 완전히 성장된 후 리프 노드에 전역 가중치를 저장하는 전략.
WbyL(Weight update by Layers): 트리의 층이 성장할 때마다 생성된 모든 노드에 전역 가중치를 저장하는 전략.
