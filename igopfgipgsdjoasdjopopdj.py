# def select_action(state):
#     state = (state - state.mean()) / (state.max() - state.min())
    
#     state = torch.from_numpy(state).float().cuda().unsqueeze(0)
    
#     if torch.isnan(state[0][0][0]):
#         print(agent.portfolio["balance"])
#         return 0
#     probs = agent(state)

#     m = Categorical(probs)
#     action = m.sample()

#     agent.recorded_actions.append(int(action.cpu().numpy()))
#     agent.saved_log_probs.append(m.log_prob(action))
    
#     if torch.isnan(probs[0][0][0]):
#         print(probs)
#         print(list(agent.parameters()))
#         print("*"*120)
#         print(state)
#         print("*"*120)
#         print(env.portfolio["balance"])
#         import sys
#         sys.exit()

#     return action
