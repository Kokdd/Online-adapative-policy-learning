import numpy as np
def maxob(ob,available_action):
    p=-np.inf
    m=0
    for j in available_action:
        if p<ob[j]:
            m=j
            p=ob[j]
    return m

def MonteCarlo_prop_score(miu,sigma,available_action):
    num=np.zeros(51)
    for i in range(0,1000):
        ob=np.zeros(50)
        for j in available_action:
            ob[j]=np.random.normal(miu[j],sigma[j])
        num[maxob(ob,available_action)]+=1
    return num/1000
