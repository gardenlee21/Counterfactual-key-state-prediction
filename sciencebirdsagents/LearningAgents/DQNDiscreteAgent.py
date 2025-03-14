import random

import numpy as np
import torch
from LearningAgents.LearningAgent import LearningAgent
from LearningAgents.Memory import ReplayMemory
from LearningAgents.RLNetwork.DQNBase import DQNBase
from SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.LevelSelection import LevelSelectionSchema
from torch.utils.tensorboard import SummaryWriter


class DQNDiscreteAgent(LearningAgent):

    def __init__(self, network: DQNBase, level_list: list,
                 replay_memory: ReplayMemory,
                 env: SBEnvironmentWrapper,
                 writer: SummaryWriter = None,
                 EPS_START=0.9,
                 level_selection_function=LevelSelectionSchema.RepeatPlay(5).select,
                 id: int = 28888, ):
        super(DQNDiscreteAgent, self).__init__(level_list=level_list, env=env, id=id, replay_memory=replay_memory, writer=writer)
        self.network = network.eval()  # the network policy
        self.replay_memory = replay_memory  # replay memory obj
        self.level_selection_function = level_selection_function
        self.state_representation_type = self.network.input_type
        self.action_type = self.network.output_type
        self.EPS_START = EPS_START
        self.input_type = self.network.input_type
        self.action_selection_mode = 'argmax'  # 'sample'

    def select_action(self, state, mode='train'):

        if mode == 'train':
            sample = random.random()
            if sample > self.EPS_START:
                with torch.no_grad():
                    if self.state_representation_type == 'image':
                        state = self.network.transform(state).unsqueeze(0).to(self.network.device)
                    elif self.state_representation_type == 'symbolic':
                        state = self.network.transform(state)
                        if hasattr(state, 'toTensor'):
                            state = state.toTensor().unsqueeze(0).to(self.network.device)
                        else:
                            state = torch.from_numpy(state).float().unsqueeze(0).to(
                                self.network.device)
                    q_values = self.network(state)

                    if self.action_selection_mode == 'sample':
                        angle = torch.Tensor(np.random.choice(range(0, q_values.size(1)), 1, p=torch.nn.Softmax(1)(
                            q_values).detach().cpu().numpy().flatten()))
                    else:
                        angle = torch.argmax(q_values, 1).to('cpu')

                    out = self.__degToShot(angle).to('cpu')
                    ################## for vta dataset ###################              
                    angle = torch.randint(105, 130, (1,)).to('cpu')
                    out = self.__degToShot(angle).to('cpu')
                    ######################################################
                    return out, angle
            else:
                q_values = torch.rand((1, self.network.outputs))
                angle = torch.argmax(q_values, 1).to('cpu')
                out = self.__degToShot(angle).to('cpu')
                ################## for vta dataset ###################              
                angle = torch.randint(105, 130, (1,)).to('cpu')
                out = self.__degToShot(angle).to('cpu')
                ######################################################
                return out, angle
        else:
            with torch.no_grad():
                if self.state_representation_type == 'image':
                    state = self.network.transform(state).unsqueeze(0).to(self.network.device)
                elif self.state_representation_type == 'symbolic':
                    state = self.network.transform(state)
                    if hasattr(state, 'toTensor'):
                        state = state.toTensor().unsqueeze(0).to(self.network.device)
                    else:
                        state = torch.from_numpy(state).float().unsqueeze(0).to(
                            self.network.device)
                q_values = self.network(state)
                if self.action_selection_mode == 'sample':
                    angle = torch.Tensor(np.random.choice(range(0, q_values.size(1)), 1, p=torch.nn.Softmax(1)(
                        q_values).detach().cpu().numpy().flatten()))
                else:
                    angle = torch.argmax(q_values, 1).to('cpu')
                out = self.__degToShot(angle) 
                return out, angle

    def __degToShot(self, deg):
        # deg = torch.argmax(q_values, 1) + 90
        deg = deg + 90 if self.network.outputs == 180 else 180
        ax_pixels = 200 * torch.cos(torch.deg2rad(deg.float())).view(-1, 1)
        ay_pixels = 200 * torch.sin(torch.deg2rad(deg.float())).view(-1, 1)
        out = torch.cat((ax_pixels, ay_pixels), 1)
        if out.size(0) == 1:
            return out[0]
        return out

    def save_q_network(self, path):
        torch.save(self.network.state_dict(), path)

    def load_q_network(self, path):
        self.network.load_state_dict(torch.load(path))


if __name__ == '__main__':
    model = DQNImage(h=224, w=224, outputs=180)
    level_list = [1, 2, 3]
    replay_memory = ReplayMemory(1000)
    env = SBEnvironmentWrapper()
    agent = DQNImageAgent(dqn=model, level_list=level_list, replay_memory=replay_memory, env=env)
    env.make(agent, speed=1, mode='head', state_representation_type='image')
    s, r, is_done, info = env.reset()
    # import matplotlib.pylab as plt
    # plt.imshow(s)
    # plt.show()
    # state = torch.rand((3, 640, 640))
    action = agent.select_action(s)
    env.step(action)
    print(action)
