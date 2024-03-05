# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.

Five Transition states / Non-terminal States including S: The starting state.

### Actions
The agent can take two actions:

R: Move right.

L: Move left.

### Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.

33.33% chance that the agent stays in its current state.

16.66% chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![rl1](https://github.com/Ishu-Vasanth/rl-policy-evaluation/assets/94154614/df4b3fc6-5102-4e8b-92de-fa14ae1b362b)

## POLICY EVALUATION FUNCTION
### Formula
![rl2](https://github.com/Ishu-Vasanth/rl-policy-evaluation/assets/94154614/5f21b9f5-da06-407a-8f26-e500563d55e0)

## Program
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
   	'''Initialize 1st Iteration estimates of state-value function(V) to zero'''
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        '''Initialize the current iteration estimates to zero'''
        V=np.zeros(len(P),dtype=np.float64)
        
        for s in range(len(P)):
        
            '''Update the value function for each state'''
            for prob,next_state,reward,done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
                
            '''Check for convergence'''
            if np.max(np.abs(prev_V-V))<theta:
                break
                
            '''Update the previous state-value function'''
            prev_V=V.copy()
        return V
```

## OUTPUT:
### Policy 1
![rl3](https://github.com/Ishu-Vasanth/rl-policy-evaluation/assets/94154614/d17e3c5e-2607-48ea-adc5-13306eb6b1e9)

### Policy 2
![rl4](https://github.com/Ishu-Vasanth/rl-policy-evaluation/assets/94154614/af663b34-575a-4a3f-8bbf-d5266dae2f88)

### Comparison
![rl5](https://github.com/Ishu-Vasanth/rl-policy-evaluation/assets/94154614/9f0d751d-4cee-4745-9153-dd16bd02428c)

### Conclusion
![rl6](https://github.com/Ishu-Vasanth/rl-policy-evaluation/assets/94154614/7db5220b-c605-47d6-b698-79ba2f2caafc)

## RESULT:
Thus, a Python program is developed to evaluate the given policy.
