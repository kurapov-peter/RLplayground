import matplotlib.pyplot as plt
import numpy as np

bandit_greedy = [.3, .4, .5, .6]


class Bandit:
    def __init__(self, greed, value):
        self.greed = greed
        self.N = 1
        self.mean = value

    def pull(self):
        return np.random.rand() + self.greed

    def update(self, reward):
        self.N += 1
        self.mean = (1 - 1. / self.N) * self.mean + reward / self.N


def chose_bandit(bandits):
    tot = sum([b.N for b in bandits])
    return np.argmax([b.mean + np.sqrt(2*np.log(tot) / b.N) for b in bandits])


def play(bandits):
    chosen = chose_bandit(bandits)
    b = bandits[chosen]
    reward = b.pull()
    b.update(reward)
    print('Chosen %s bandit, got %s reward' % (chosen, reward))
    return reward


N = 10000
if __name__ == '__main__':
    bandits = [
        [Bandit(g, 0) for g in bandit_greedy],
        [Bandit(g, 0) for g in bandit_greedy]
    ]

    for b in bandits:
        rewards = []
        for i in range(N):
            rewards.append(play(b))
        avg = np.cumsum(rewards) / (np.arange(N) + 1)
        plt.plot(avg)

    plt.xscale('log')
    plt.show()
