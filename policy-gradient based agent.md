### **Policy Gradient based Agent**

###### Q 러닝은 현재 state에서 가장 높은 reward를 받는 액션을 취할 수 있도록 했다면, Policy-gradient는 long run에 대해 최적의 reward를 획득 할 수 있도록 액션을 취하는 데 초점이 맞추어져 있다.

###### 이 문제를 설명한 환경이 마르코프 결정 과정(Markov Decision Process, MDP)이다. 이 환경은 주어진 액션으로 reward와 next state만 단순히 제공하는 것이 아니다. 여기서는 reward도 상태와 그 상태에서 agent가 취하는 액션에 대한 조건부 확률이다. 이러한 dynamics는 temporal하며, 시간적으로 delay될 수 있다.

###### 보다 형식적으로 설명하자면, 다음과 같이 MDP를 설명할 수 있다. MDP가 agent가 어떤 시간에서 상태 s를 경험한다하면, 그 이후 모든 가능한 상태들의 집합을 S라 하자. MDP는 S로 구성되어있다. 그리고, agent가 어떤 시간에서 액션 a를 취한다면, 그 이후 모든 가능한 액션들의 집합을 A라 하자. state-action pair (s, a)가 있을 때, 그래서 새로운 state s'으로 transition되었다면, T(s,a)로 정의되고, reward는 R(s,a)로 정의된다. 이런식으로, MDP의 어떤 시간에서, agent는 state s가 주어지면, 액션 a를 취하고, 새로운 state s'과 reward r을 받는다.

###### 일견 간단해보이지만, 거의 대부분의 task들이 MDP로 설명될 수 있다. 예를 들어, 문을 여는 task를 생각해보자. state는 문의 방향, 우리 몸과 문의 위치일 것이다. 액션은 우리가 할 수 있는 모든 움직임이 될 것이고, reward는 문이 제대로 열렸는지의 여부일 것이다. 문쪽으로 걸어가는 것은 이 문제를 풀 때 필수적인 액션이라 할 수 있다. 그러나 실제적으로 문을 열기위한 액션은 아니므로 reward는 받을 수 없다. 따라서 agent는 궁극적으로 reward를 받도록 이끄는 액션을 하도록 학습될 필요가 있으므로 temporal dynaimics를 도입해야한다.

###### reward over time을 고려하기 위해, agent는 한번에 하나이상의 experience를 가지고 업데이트되어야한다. 이를 구현하기 위해서는 experiece들을 버퍼에 모아 때때로 한번에 업데이트할 것이다. 이러한 sequences of experience는 'rollout' 또는 'experience trace'라고 한다. 하지만 이러한 rollout들을 그냥쓰는게 아니라 discount factor로 적절하게 조정하고 사용한다.

###### 직관적으로 이는 각 액션이 즉각적으로 들어오는 reward만이 아니라 후에 들어오는 reward에도 어느정도 영향을 미치게끔한다. 우리는 loss function에 이 수정된 reward를 advantage의 추정치로 쓸 것이다.