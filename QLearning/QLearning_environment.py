import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

#Used to make the graphs 'pretty'
style.use('ggplot')

###############################################################################################
#------------------------------------Constants------------------------------------------------#
SIZE = 10
HM_EPISODES = 25_000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

#Exploration settings
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 1_000

start_q_table = None #or filename

#Learning parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
STEPS = 200

#Definitions to go into the dictionary
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
d = {1: (255, 175, 0), 
	 2: (0, 255, 0),
	 3: (0, 0, 255)} #using BGR colours
#---------------------------------------------------------------------------------------------#
###############################################################################################

###############################################################################################
#---------------------------------Classes/Functions-------------------------------------------#
class Blob:
	def __init__(self):
		self.x = np.random.randint(0, SIZE)
		self.y = np.random.randint(0, SIZE)

	def __str__(self):
		return f"{self.x}, {self.y}"

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)

	#Action space for use in QLearning
	def action(self, choice):
		if choice == 0:
			self.move(x = 1, y = 1)
		elif choice == 1:
			self.move(x = -1, y = -1)
		elif choice == 2:
			self.move(x = -1, y = 1)
		elif choice == 3:
			self.move(x = 1, y = -1)
		elif choice == 4:
			self.move(x = 1, y = 0)
		elif choice == 5:
			self.move(x = -1, y = 0)
		elif choice == 6:
			self.move(x = 0, y = 1)
		elif choice == 7:
			self.move(x = 0, y = -1)


	def move(self, x = False, y = False):
		#Basic movement
		if not x:
			self.x += np.random.randint(-1, 2) #-1, 0, or 1
		else:
			self.x += x

		if not y:
			self.y += np.random.randint(-1, 2) #-1, 0, or 1
		else:
			self.y += y

		#Constraining within the SIZE parameter
		if self.x < 0:
			self.x = 0
		elif self.x > SIZE - 1:
			self.x = SIZE - 1
		if self.y < 0:
			self.y = 0
		elif self.y > SIZE - 1:
			self.y = SIZE - 1

#Initialising the q table
def InitialiseQ():
	if start_q_table is None:
		q_table = {}
		for x1 in range(-SIZE + 1, SIZE):
			for y1 in range(-SIZE + 1, SIZE):
				for x2 in range(-SIZE + 1, SIZE):
					for y2 in range(-SIZE + 1, SIZE):
						q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-9 , 0 ) for i in range(8)]
	#Loading a pre-trained q table					
	else:
		with open(start_q_table, "rb") as f:
			q_table = pickle.load(f)

	return q_table

#Begin training
def TrainQ(epsilon):
	episode_rewards = []
	for episode in range(HM_EPISODES):
		player = Blob()
		food = Blob()
		enemy = Blob()

		if episode % SHOW_EVERY == 0:
			print(f"on #{episode}, epsilon: {epsilon}")
			print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
			show = True
		else:
			show = False

		episode_reward = 0
		for i in range(STEPS):
			obs = (player - food, player - enemy)
			if np.random.random() > epsilon:
				action = np.argmax(q_table[obs])
			else:
				action = np.random.randint(0, 8)

			player.action(action)

			### Potential for enemy.move() and food.move()
			enemy.move()
			food.move()

			#Setting the rewards for the various outcomes
			if player.x == enemy.x and player.y == enemy.y:
				reward = -ENEMY_PENALTY
			elif player.x == food.x and player.y == food.y:
				reward = FOOD_REWARD
			else:
				reward = -MOVE_PENALTY

			#Finding the current, and max future q values
			new_obs = (player - food, player - enemy)
			max_future_q = np.max(q_table[new_obs])
			current_q = q_table[obs][action]

			#Setting the new q values
			if reward == FOOD_REWARD:
				new_q = FOOD_REWARD
			elif reward == -ENEMY_PENALTY:
				new_q = -ENEMY_PENALTY
			else:
				new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

			#Update the q table
			q_table[obs][action] = new_q

			#Showing the environment
			if show:
				env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)
				env[food.y][food.x] = d[FOOD_N]
				env[player.y][player.x] = d[PLAYER_N]
				env[enemy.y][enemy.x] = d[ENEMY_N]

				img = Image.fromarray(env, "RGB")
				img = img.resize((300, 300))

				cv2.imshow("", np.array(img))
				if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
					if cv2.waitKey(500) & 0xFF == ord("q"):
						break
				else:
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break

			episode_reward += reward
			if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
				break

		episode_rewards.append(episode_reward)
		epsilon *= EPS_DECAY

	return episode_rewards

#Graphing things:
def GraphAndAnalyseQ(SAVE_Q_TABLE):
	moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode = "valid")

	plt.plot([i for i in range(len(moving_avg))], moving_avg)
	plt.ylabel(f"Reward {SHOW_EVERY}")
	plt.xlabel("episode #")
	plt.show()

	if SAVE_Q_TABLE == True:
		with open(f"Qtables/qtable-{int(time.time())}.pickle", "wb") as f:
			pickle.dump(q_table, f)
#---------------------------------------------------------------------------------------------#
###############################################################################################

###############################################################################################
#----------------------------------Functional Code--------------------------------------------#
q_table = InitialiseQ()
episode_rewards = TrainQ(0.9)
GraphAndAnalyseQ(False)
#---------------------------------------------------------------------------------------------#
###############################################################################################





