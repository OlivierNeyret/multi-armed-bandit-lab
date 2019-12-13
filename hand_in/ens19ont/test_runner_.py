import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))

import bandit

def simulate():
    results = []
    for _ in range(0, 20):
        bandit_reward = bandit.simulator.simulate(bandit.bandit)
        ref_bandit_reward = bandit.simulator.simulate(bandit.ref_bandit)

        print('Expected values of the different arms:')
        print(bandit.bandit.expected_values)
        print('Frequencies')
        print(bandit.bandit.frequencies)

        ref_plus_bonus = ref_bandit_reward * 1.2 
        result = 0
        if (bandit_reward > ref_plus_bonus):
            result = 1
        results.append(result)
    return results

def test_performance():
    # assert True
    s = sum(simulate())
    print(s)
    assert s > 17

test_performance()
