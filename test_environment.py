from Environment2 import Environment
import unittest

class TestCases(unittest.TestCase):
        
    def test_empty_sell_rejection(self):
        env = Environment()
        env.reset()
        init_balance = env.portfolio["balance"]
        try:
            state, reward, done = env.step({
                "buy":0,
                "sell":3
            })
            self.assertTrue(False, "Disabled Selling With Empty Porflio")
        except AssertionError:
            self.assertTrue(True)

    def test_buying(self):
        env = Environment()
        env.reset()

        init_balance = env.portfolio["balance"]
        state, reward, done = env.step({
            "buy": 3,
            "sell": 0
        })
        self.assertLess(env.portfolio["balance"], init_balance, "Buying Does Decrease Account Balance")

    def test_selling(self):
        env = Environment()
        env.reset()
        
        
        state, reward, done = env.step({
            "buy": 3,
            "sell": 0
        })
        init_balance = env.portfolio["balance"]
        
        state, reward, done = env.step({
            "buy": 0,
            "sell": 2
        })
        after_balance = env.portfolio["balance"]
        self.assertGreater(after_balance, init_balance, "Selling Does Increase Account Balance")

    def test_rewards(self):
        env = Environment()
        env.reset()

        state, reward, done = env.step({
            "buy": 3,
            "sell": 0
        })
        
        sell_price = env._get_state(False)[0].iloc[-1]["close"]
        state, reward, done = env.step({
            "buy": 0,
            "sell": 3
        })

        
        
        self.assertEqual(reward, sell_price *3, "Rewards are Estimated Correctly")


if __name__ == "__main__":
    unittest.main()