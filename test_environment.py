from Environment import Environment
import unittest

class TestCases(unittest.TestCase):
        
    def test_empty_sell_rejection(self):
        env = Environment(2)
        env.load_stock()
        init_balance = env.agent_balance
        state, reward, done = env.step({
            "buy":0,
            "sell":3
        })
        self.assertEqual(env.agent_balance, init_balance, "Disabled Selling With Empty Porflio")

    def test_buying(self):
        env = Environment(2)
        env.load_stock()
        init_balance = env.agent_balance
        state, reward, done = env.step({
            "buy": 3,
            "sell": 0
        })
        self.assertLess(env.agent_balance, init_balance, "Buying Does Decrease Account Balance")

    def test_selling(self):
        env = Environment(2)
        env.load_stock()
        state, reward, done = env.step({
            "buy": 3,
            "sell": 0
        })
        init_balance = env.agent_balance
        state, reward, done = env.step({
            "buy": 0,
            "sell": 2
        })
        self.assertGreater(env.agent_balance, init_balance, "Selling Does Increase Account Balance")

    def test_rewards(self):
        k_shares = 3
        env = Environment(2)
        env.load_stock()
        state, reward, done = env.step({
            "buy": k_shares,
            "sell": 0
        })
        purchase_price = env._latest_price(n=-1)
        state, reward, done = env.step({
            "buy": 0,
            "sell": k_shares
        })
        sell_price = env._latest_price(n=-1)
        est_reward = (sell_price - purchase_price)  * 3
        #print("Returned: {}".format(reward))
        #print("Estimated: {}".format(est_reward))
        self.assertEqual(reward, est_reward, "Rewards are Estimated Correctly")

    def test_early_iteration_stop(self):
        est_days = 100
        env = Environment(1)
        env.load_stock()
        for _ in range(est_days):
            _, _, done = env.step({"buy":0, "sell":0})
            if done: break
        self.assertEqual(env.step_num, est_days, "Failed Early Iteration Stopping")

if __name__ == "__main__":
    unittest.main()