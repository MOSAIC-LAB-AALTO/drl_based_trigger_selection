# Description

Project for selecting triggers to enable network slice mobility using Reinforcement Learning (RL) as well as Deep Reinforcement Learning (DRL) techniques.

# Requirement
1. python > 3
2. gym
3. tensorflow
4. keras
5. rekars-rl

### Prerequisites


```
Python 3.6.7
```
```
gym
```
```
tensorflow
```
```
keras
```
```
rekars-rl
```



# How to use it

1. Select the type of DRL algorithm to use
```
python3 dev_test.py --train [dqn, fix_dqn, double_dqn]

```
2. Try the trained model
```
python3 dev_test.py --train [dqn, fix_dqn, double_dqn] --observe [test number]

```


# Possible improvement

1. Implement new DRL algorithms
2. State representation
3. Scalability


## Authors
* **Rami Akrem Addad** - *Initial work, system level optimization, modeling designer* - [ramy150](https://github.com/ramy150)

