import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from Action import action
from Observation import observation
from Functions import MonteCarlo_prop_score


def Run(K,snr,X):

    SNR = snr
    T = 10000
    gamma = 0.01
    # initialization
    A = [action(T) for i in range(0, K)]
    Best_reward = -10
    Best_arm=0
    Best_arm_estimator=0
    for a in range(0, K):
        A[a].reward = np.random.normal(0, SNR)
        if Best_reward<A[a].reward:
            Best_reward = A[a].reward
            Best_arm=a

    Best_reward_estimator = np.zeros(T + 1)
    available_action = [i for i in range(0, K)]

    k = K
    tao = np.zeros((T + 2, k))
    miu_reward_estimator = np.zeros((T + 2, k))
    sigma_estimator = np.zeros((T + 2, k))

    # First Observation
    for x in range(0,X):
        for a in range(0, k):
            A[a].reward_mean = observation(A[a]).value
            A[a].pro_score[1] = 1 / k

    for t in range(1, T + 1):

        #calculate regret
        for a in available_action:   Best_reward_estimator[t]+=A[a].pro_score[t]*A[a].reward

        # pull arm
        a_t = np.random.multinomial(1, [A[j].pro_score[t] for j in available_action])
        a_t = available_action[np.argwhere(a_t == 1)[0, 0]]

        # observe and update sample averages and counts
        ob = observation(A[a_t])
        re_past = A[a_t].reward_mean
        A[a_t].update_rm_n(ob)

        # update sampling distribution
        for a in available_action:
            tao[t, a] = re_past + (ob.value - re_past) / A[a].pro_score[t] if a == a_t else A[
                a].reward_mean

            s1 = sum(np.sqrt(A[a].pro_score[1:t + 1]) * tao[1:t + 1, a])
            s2 = sum(np.sqrt(A[a].pro_score[1:t + 1]))
            miu_reward_estimator[t, a] = s1 / s2

            s3 = sum([A[a].pro_score[s] * ((tao[s, a] - miu_reward_estimator[t, a]) ** 2 + 1) for s in range(1, t + 1)])
            sigma_estimator[t, a] = s3 / (s2 * s2)

        # arm elimination
        rem = []
        for a in available_action:
            p = 1
            for a1 in available_action:
                if a != a1:
                    p1 = norm.cdf((miu_reward_estimator[t, a] - miu_reward_estimator[t, a1]) / (
                            (sigma_estimator[t, a] ** 2 + sigma_estimator[t, a1] ** 2) ** 0.5))
                    p = min(p, p1)

            if p < 1 / T:
                rem.append(a)
                k -= 1

        for a in rem:
            available_action.remove(a)

        # compute propensity scores
        Prop_score=MonteCarlo_prop_score(miu_reward_estimator[t], sigma_estimator[t], available_action)
        for a in available_action:
            A[a].pro_score[t + 1] = Prop_score[a]
            A[a].pro_score[t + 1] = (1 - gamma) * A[a].pro_score[t + 1] + gamma / k

        #break after finding the best arm
        if len(available_action) == 1:
            Best_arm_estimator = available_action[0]
            print('END!')
            print('Best arm is the ' + str(Best_arm_estimator) + 'th arm.')
            break

        if len(available_action) == 0:
            print('end!')
            Best_arm_estimator = rem[0]
            for a in rem:
                Best_arm_estimator = a if A[a].reward_mean > A[Best_arm_estimator].reward_mean else Best_arm_estimator
            print('Best arm is the ' + str(Best_arm_estimator) + 'th arm.')
            break

    if t == 10000:
        print('End!')
        Best_arm_estimator = available_action[0]
        for a in available_action:
            Best_arm_estimator = a if A[a].reward_mean > A[Best_arm_estimator].reward_mean else Best_arm_estimator
        print('Best arm is the ' + str(Best_arm_estimator) + 'th arm.')

    print('K=' + str(K) + '  t=' + str(t))
    cumulative_regret=sum(Best_reward - Best_reward_estimator[1:t + 1]) / t
    Regret=Best_reward-A[Best_arm_estimator].reward
    print('cumulative regret='+str(cumulative_regret)+'    Arm* regret='+str(Regret))
    return [cumulative_regret,Regret,t]

Cu_Regret=np.zeros(10)
Regret=np.zeros(10)
stop_time=np.zeros(10)
for k in range(0,1):
    for i in range(0,1):
        Cu_Re,Re,st=Run(5*k+5,0.125,1)
        Cu_Regret[k]+=Cu_Re/64
        stop_time[k]+=st/64
        Regret[k]=Re/64
    print('Regret Average is '+str(Cu_Regret[k]))

K=[5,10,15,20,25,30,35,40,45,50]
FinalData = {"Cu_Regret":Cu_Regret[:], "stopping_time":stop_time[:],"Regret":Regret[:]}
df=pd.DataFrame(FinalData)
df.to_excel('./FinalData.xlsx',index=False)
