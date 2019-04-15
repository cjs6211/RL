import gym
import numpy as np
import tensorflow as tf
import threading
import random
import time
from skimage.color import rgb2gray
from skimage.transform import resize

#Global variable
global episode
episode = 0
EPISODES = 8000000

#gym env
env_name = "BreakoutDeterministic-v4"

def get_copy_var_ops(*, dest_scope_name="local", src_scope_name="global"):
    #copy variables src_scope to dest_scope_name
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return  op_holder

def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), output_shape=(84, 84), mode='constant') * 255)
    return processed_observe

#Advantage Actor Critic agent
class Model:
    def __init__(self, action_size, value_size, net_name, actor_lr, critic_lr, sess):
        self.sess = sess
        self.net_name = net_name
        self.action_size = action_size
        self.value_size = value_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        with tf.variable_scope(self.net_name):
            self.action = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name="action")
            self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="advantage")
            self.input = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name="input")
            self.discounted_prediciton = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="discounted_prediction")

            self.conv1 = tf.layers.conv2d(name="CONV1", inputs=self.input, filters=32, kernel_size=[8, 8], strides=[4, 4], use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.layer1 = tf.nn.relu(self.conv1)
            self.conv2 = tf.layers.conv2d(name="CONV2", inputs=self.layer1, filters=64, kernel_size=[4, 4], strides=[2, 2], use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.layer2 = tf.layers.flatten(tf.nn.relu(self.conv2))
            self.fc = tf.layers.dense(name="DENSE", inputs=self.layer2, units=512, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.policy = tf.layers.dense(name="POLICY", inputs=self.fc, units=self.action_size, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.policy = tf.nn.softmax(self.policy)
            self.critic = tf.layers.dense(name="CRITIC", inputs=self.fc, units=self.value_size, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.action_prob = tf.reduce_sum(self.action * self.policy, axis=1)
            self.policy_loss = -tf.reduce_sum(tf.log(self.action_prob + 1e-8) * self.advantage)

            # for exploration
            self.extra_loss = tf.reduce_sum(tf.reduce_sum(self.policy * tf.log(self.policy + 1e-8), axis=1))
            self.policy_loss = self.policy_loss + 0.1 * self.extra_loss

            self.critic_loss = tf.reduce_mean(tf.square(self.discounted_prediciton - self.critic))

            #train
            self.policy_train = tf.train.AdamOptimizer(learning_rate=self.actor_lr).minimize(self.policy_loss)
            self.critic_train = tf.train.AdamOptimizer(learning_rate=self.critic_lr).minimize(self.critic_loss)

    def get_policy(self, state):
        return self.sess.run(self.policy, feed_dict={self.input:state})

    def get_Qpred(self, state):
        return self.sess.run(self.critic, feed_dict={self.input:state})

    def train_policy(self, states, actions, advantages):
        self.sess.run(self.policy_train, feed_dict={self.input:states, self.action:actions, self.advantage:advantages})

    def train_critic(self, states, discounted_prediction):
        self.sess.run(self.critic_train, feed_dict={self.input:states, self.discounted_prediciton:discounted_prediction})

    def init_model(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess, discount_factor, summary_ops, net_name):
        threading.Thread.__init__(self)

        self.net_name = net_name
        self.action_size = action_size
        self.value_size = 1
        self.state_size = state_size
        self.model = model
        self.sess = sess
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placehoders, self.update_ops, self.summary_writer] = summary_ops

        self.states, self.actions, self.rewards = [], [], []

        #copy global weights to local
        self.local_model = Model(action_size=self.action_size, value_size=self.value_size, net_name=self.net_name,
                                 actor_lr=2.5e-4, critic_lr=2.5e-4, sess=self.sess)
        self.local_model.init_model()
        self.copy_ops = get_copy_var_ops(dest_scope_name=self.net_name, src_scope_name="global")
        self.sess.run(self.copy_ops)

        self.avg_p_max = 0
        self.avg_loss = 0

        #update frequency
        self.t_max = 20
        self.t = 0

    def run(self):
        global episode
        env = gym.make(env_name)

        step = 0

        while episode<EPISODES:
            done = False
            dead = False

            score, start_life = 0, 5
            observe = env.reset()
            next_observe = observe

            for _ in range(random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = env.step(1) #1:stop 2:left 3:right

            state = pre_processing(next_observe=next_observe, observe=observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                if self.net_name == "local_0":
                    env.render()
                step += 1
                self.t += 1
                observe = next_observe
                action, policy = self.get_action(history=history)
                #print("action:{}, policy:{}".format(action, policy))

                if action == 0 :
                    real_action = 1
                elif action == 1 :
                    real_action = 2
                else:
                    real_action = 3

                if dead: #launch the ball when die
                    action, real_action, dead = 0, 1, False

                #run step
                next_observe, reward, done, info = env.step(real_action)

                #preprocessing state at each time step
                next_state = pre_processing(next_observe=next_observe, observe=observe)
                next_state = np.reshape([next_state], [1, 84, 84, 1])
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                #maximum of policy
                self.avg_p_max += np.amax(self.model.get_policy(np.float32(history/255.0)))

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                #store sample
                self.append_sample(history=history, action=action, reward=reward)

                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                #train after t_max or done
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t=0

                if done:
                    episode += 1
                    print("thread: {} episode: {} score: {} step: {} avg_p_max: {}".format(self.net_name,
                                                                                           episode, score, step, self.avg_p_max))

                    stats = [score, self.avg_p_max/float(step), step]

                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placehoders[i]:float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode+1)
                    self.avg_p_max=0
                    self.avg_loss=0
                    step=0

    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(rewards=self.rewards, states=self.states, done=done)
        states = np.zeros((len(self.states), 84, 84, 4))
        for i in range(len(self.states)):
            states[i] = self.states[i]
        states = np.float32(states/255.0)
        values = self.model.get_Qpred(state=states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        advantages = np.reshape(advantages, [self.t, 1])
        discounted_prediction = np.reshape(discounted_prediction, newshape=[self.t, 1])

        self.model.train_policy(states=states, actions=self.actions, advantages=advantages)
        self.model.train_critic(states=states, discounted_prediction=discounted_prediction)


        self.states, self.rewards, self.actions = [], [], []

    def discounted_prediction(self, rewards, states, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.model.get_Qpred(states[-1]/255.0)[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add

        return discounted_prediction

    def get_action(self, history):
        history = np.float32(history/255.0)
        policy = self.local_model.get_policy(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def update_local_model(self):
        #copy_ops = get_copy_var_ops(dest_scope_name=self.net_name, src_scope_name='global')
        self.sess.run(self.copy_ops)
        #print("Thread {} is updated".format(self.net_name))


class A3CAgent:
    def __init__(self, action_size, sess):
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.value_size = 1

        # hyperparameters
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 1.0e-4
        self.critic_lr = 1.0e-4

        #no of threads
        self.threads = 8

        #generate policy and critic networks
        self.sess = sess
        self.model = Model(action_size=self.action_size, value_size=self.value_size, net_name="global",
                           actor_lr=2.5e-4, critic_lr=2.5e-4, sess=self.sess)
        self.model.init_model()
        #setting tensorboard
        self.summary_placeholder, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    def train(self):
        #make local agents of number of threads
        agents = [Agent(action_size=self.action_size, state_size=self.state_size, model=self.model, sess=self.sess,
                        discount_factor=self.discount_factor,
                        summary_ops=[self.summary_op, self.summary_placeholder, self.update_ops, self.summary_writer],
                        net_name="local_"+str(i)) for i in range(self.threads)]

        #start each thread
        for agent in agents:
            time.sleep(1)
            agent.start()

        #save model at every 10 min
        while True:
            time.sleep(60*10)
            self.save_model("./save_model/breakout_A3C")


    def load_model(self, name):
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=name+"_model.ckpt")

    def save_model(self, name):
        saver = tf.train.Saver()
        saver.save(sess=self.sess, save_path=name+"_model.ckpt")

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.0)
        episode_avg_max_q = tf.Variable(0.0)
        episode_duration = tf.Variable(0.0)

        tf.summary.scalar('Total reward/episode', episode_total_reward)
        tf.summary.scalar('Average Max prob/episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]

        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]

        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op



if __name__=="__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    global_agent = A3CAgent(action_size=3, sess=sess)
    global_agent.train()