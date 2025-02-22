<h1>LunarLander-v2</h1>
<hr/>

<h2>Requirement</h2>
<hr/>
<h4>Libraries:</h4>

- gym
- Box2D
- tensorflow
- numpy
- keras

<h2>LunarLander-v2</h2>
<hr/>
<h4>Actions:</h4>

- Do nothing.
- Fire left orientation engine.
- Fire main engine.
- Fire right orientation engine.

<h4>Observation Space:</h4>

- Lander horizontal coordinate.
- Lander vertical coordinate.
- Lander horizontal speed.
- Lander vertical speed.
- Lander angle.
- Lander angular speed.
- Bool: 1 if first leg has contact, else 0.
- Bool: 1 if second leg has contact, else 0.

<h4>Rewards:</h4>

- Moving from the top of the screen to the landing pad and coming to rest: 100-140 points.
- The lander crashes: -100 points.
- Comes to rest: +100 points.
- Each leg with ground contact: +10 points.
- Firing the main engine: -0.3 points.
- Firing the side engine: -0.03 points.

<h4>Episode Termination:</h4>

- The lander crashes.
- The lander gets outside of the viewport.
- Lander comes to rest.

<h2>Deep Q-Network (DQN)</h2>

- Number of episodes: 1100
- Discount Factor: 0.99
- Epsilon: 1.0
- Epsilon min: 0.01
- Epsilon_decay: 0.98
- Learning rate: 0.001
- Number steps update target network: 10
- Batch_size: 64
