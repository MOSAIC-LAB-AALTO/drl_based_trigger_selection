# Description

Project for selecting triggers to enable network slice mobility using Reinforcement Learning (RL) as well as Deep Reinforcement Learning (DRL) techniques.

# Requirement
1. python > 3
2. pytorch

### Prerequisites


```
Python 3.6.7
```
```
Pytorch
```




# How to use it

1. Select the type of DRL algorithm to use
```
python3 main_dqn.py/main_a2c.py --train [dqn/a2c]

```
2. Try the trained model
```
python3 main_dqn.py/main_a2c.py --train [dqn/a2c] --observe [test number]

```


# Possible improvement

1. Implement new DRL algorithms (PPO)
2. State representation and modeling
3. Scalability


## Authors
* **Rami Akrem Addad** - *Initial work, system level optimization, modeling designer* - [ramy150](https://github.com/ramy150)

