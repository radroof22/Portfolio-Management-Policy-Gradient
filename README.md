# Portfolio Management Policy Gradient
The goal of this project was to see how effective a reinforcement learning type training model would be for financial management. Normal models rely on linear regression and predictions of the next price of a stock. Instead, this project sought to determine if policy gradient, a type of reinforcement model, may be effective for portfolio management.

## Awards
* 3rd Place at Delware Valley Science Fair for Computer Science in 10th Grade Division

## Technology Overview
- **Pytorch** for developing the policy gradient neural network
- **Pandas** for dataframe manipulation of S&P 500
- **Numpy** for mathematical functions

## Workflow
1. Load Stock from the S&P 500
2. Reset the Agent to have $100,000 balance and 0 shares of any stocks
3. Provide state of the current stock prices of the day to the model
4. Execute order if applicable based on the buy, hold, sell signals from the model for each stock
5. Provide reward to the model for if portfolio value has increased or profit has been made (greater reward if profit increase)

## Observation Space
* High, Low, Open, Close, Volume
* Values returned as differences from the previous value

## Reward Calculation
`balance + sum(number of shares * share price) for each stock`

## Model Overview
* Learning Rate (how aggressive model optimizations are) = 1e-6
* Gamme (how to value previous timestep data when making current decision) = 0.99
* Optimizer = Adam

## Data
![Final Balance of Agent v. Risk Free Return](https://user-images.githubusercontent.com/23004551/119423041-6219a780-bcd0-11eb-8823-b0e1fb78207c.png)
As seen here, although the model occasionally has a negative return, it has more positive returns, which are all greater than the negative returns the model obtains

![Rate of Return Frequency](https://user-images.githubusercontent.com/23004551/119423044-6219a780-bcd0-11eb-822e-76dfd871ce97.png)
As seen here, the rate of positive returns are much more likely that the minimal risk of negative returns observered based on the testing procedure

## Data Analysis
* **Null Hypothesis**: The machine learning model will not be significantly better than the risk free return
* **Anti-Hypothesis**: The use of machine learning models on stock data will outperform the risk-free return on a 5 year scale
* Assessed risk-return ratio by calculating **Sharpe's Ratio** and comparing to risk-free investment along with t-test
* **P-value = 1.1%** < 5%
* Sharpes Ratio is calculated to be 0.3
