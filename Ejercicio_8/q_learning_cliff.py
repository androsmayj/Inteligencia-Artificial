import random
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ACTIONS = ['N', 'E', 'S', 'W']
ACTION_EFFECTS = {
    'N': (-1, 0),
    'E': (0, 1),
    'S': (1, 0),
    'W': (0, -1)
}
ARROWS = {'N': '↑', 'E': '→', 'S': '↓', 'W': '←'}

class CliffWalkingEnv:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.cliff_states = [(3, c) for c in range(1, 11)]
        self.reset()

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        dr, dc = ACTION_EFFECTS[action]
        r, c = self.state
        
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        next_pos = (nr, nc)

        if next_pos in self.cliff_states:
            self.state = self.start_state
            return self.state, -100, False
        elif next_pos == self.goal_state:
            self.state = next_pos
            return self.state, -1, True
        else:
            self.state = next_pos
            return self.state, -1, False

class QLearningAgent:
    def __init__(self, actions, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def get_q(self, state, action):
        return self.Q.get(state, {}).get(action, 0.0)

    def select_action(self, state, greedy=False):
        eps = 0.0 if greedy else self.epsilon
        if random.random() < eps:
            return random.choice(self.actions)
        
        best_action = self.actions[0]
        max_val = self.get_q(state, best_action)
        for a in self.actions[1:]:
            val = self.get_q(state, a)
            if val > max_val:
                max_val = val
                best_action = a
        return best_action

    def update(self, s, a, r, s_prime, done):
        current_q = self.get_q(s, a)
        if done:
            target = r
        else:
            max_q_next = max(self.get_q(s_prime, a_p) for a_p in self.actions)
            target = r + self.gamma * max_q_next
        
        if s not in self.Q:
            self.Q[s] = {act: 0.0 for act in self.actions}
        self.Q[s][a] = current_q + self.alpha * (target - current_q)

def run_experiment():
    env = CliffWalkingEnv()
    agent = QLearningAgent(ACTIONS, alpha=0.5, gamma=1.0, epsilon=0.1)
    
    num_episodes = 500
    max_steps_per_episode = 200
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for _ in range(max_steps_per_episode):
            action = agent.select_action(state, greedy=False)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            if done:
                break
                
        rewards_per_episode.append(total_reward)

    state = env.reset()
    path = [state]
    actions_taken = []
    greedy_return = 0
    done = False
    
    while not done and len(path) <= 20:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done = env.step(action)
        greedy_return += reward
        actions_taken.append(action)
        state = next_state
        path.append(state)

    return agent, rewards_per_episode, path, actions_taken, greedy_return

if __name__ == "__main__":
    agent, rewards, path, actions_taken, greedy_return = run_experiment()

    print(f"Semilla utilizada: {SEED}")
    print(f"Retorno greedy obtenido: {greedy_return}")
    print(f"Número de pasos tomados: {len(actions_taken)}")
    print(f"Secuencia de acciones: {' -> '.join(actions_taken)}")
    print(f"Camino recorrido: {path}\n")

    print("--- POLÍTICA GREEDY APRENDIDA (CUADRÍCULA) ---")
    grid_str = []
    for r in range(4):
        row_cells = []
        for c in range(12):
            st = (r, c)
            if st == (3, 0):
                row_cells.append("  S  ")
            elif st == (3, 11):
                row_cells.append("  G  ")
            elif st in [(3, i) for i in range(1, 11)]:
                row_cells.append("  C  ")
            else:
                best_act = agent.select_action(st, greedy=True)
                row_cells.append(f"  {ARROWS[best_act]}  ")
        print("".join(row_cells))

    moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='gray', label='Recompensa cruda por episodio')
    plt.plot(range(9, 500), moving_avg, color='blue', label='Media móvil (10 episodios)')
    plt.axhline(-13, color='red', linestyle='--', label='Óptimo Greedy (-13)')
    plt.title('Curva de Aprendizaje de Q-Learning en Cliff Walking (ε = 0.1, α = 0.5)')
    plt.xlabel('Episodio')
    plt.ylabel('Retorno Acumulado')
    plt.ylim(-150, 0)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()