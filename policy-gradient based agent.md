**Policy Gradient based Agent**

Q 러닝은 현재 state에서 가장 높은 reward를 받는 액션을 취할 수 있도록 했다면, Policy-gradient는 long run에 대해 최적의 reward를 획득 할 수 있도록 액션을 취하는 데 초점이 맞추어져 있다.

이 문제를 설명한 환경이 마르코프 결정 과정(Markov Decision Process, MDP)이다. 이 환경은 주어진 액션으로 reward와 next state만 단순히 제공하는 것이 아니다. 여기서는 reward도 상태와 그 상태에서 agent가 취하는 액션에 대한 조건부 확률이다. 이러한 dynamics는 temporal하며, 시간적으로 delay될 수 있다.

보다 형식적으로 설명하자면, 다음과 같이 MDP를 설명할 수 있다. MDP가 