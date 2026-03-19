import numpy as np
from tqdm import trange
from pydmd.utils import pseudo_hankel_matrix

def generate_imitation_training_data(agent, env, n_episodes, max_ep_length=None, seed=23):
    observations = []
    actions = []
    rewards = np.zeros(n_episodes)
    for iEpisode in trange(n_episodes):
        ep_obs = []
        ep_acts = []
        env.seed(seed+iEpisode)
        obs = env.reset()
        ep_reward = 0
        for i in range(max_ep_length):
            ep_obs.append(obs) # store the observation
            action, _ = agent.predict(obs)
            obs, step_reward, done, info = env.step(action)
            ep_acts.append(action) # store corresponding action

            ep_reward += step_reward

            if done:
                break
                
        env.close()
    
        observations.append(ep_obs)
        actions.append(ep_acts)
        rewards[iEpisode] = ep_reward #/ max_ep_length

    return {'observations': observations, 'actions': actions, 'rewards': rewards}


def concatenate_episodes(episodes_list, actions_list, n_actions, n_time_delays=None, discrete = True):

    X1 = []
    X2 = []

    U = []

    for iEpisode, X in enumerate(episodes_list):

        # X is a list of observations
        if type(X) == list:
            X = np.vstack(X).T

        u_ep = actions_list[iEpisode]
        if type(u_ep) == list:
            u_ep = np.vstack(u_ep).T

        # convert u_ep into a one hot encoded vector
        if discrete:
            u_ep_encoded = np.eye(n_actions)[u_ep.squeeze()].T
        else:
            u_ep_encoded = u_ep

        if n_time_delays > 1:
            # Hankelization
            X = pseudo_hankel_matrix(X, n_time_delays)
            #u_ep_encoded = pseudo_hankel_matrix(u_ep_encoded, n_time_delays)
            u_ep_encoded = u_ep_encoded[:, :X.shape[-1]]

        # split into X and Xprime (using this method to avoid having erroneous T_end, T_start pairs)
        x1_ep = X[:, :-1]
        x2_ep = X[:, 1:]

        X1.append(x1_ep)
        X2.append(x2_ep)

        U.append(u_ep_encoded[:, :-1])

    X1 = np.hstack(X1)
    X2 = np.hstack(X2)
    U = np.hstack(U)

    return X1, X2, U