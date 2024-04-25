import numpy as np

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnglishAuctionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    class BiddingObject:
        """
        Represents a bidding object in an English auction.

        Attributes:
            stats (numpy.ndarray): An array of size 3 representing the statistics of the bidding object.
        """
        def __init__(self, id : int) -> None:
            self.id = id
            self.stats = np.random.rand(3)
            self.current_bid = 0
            self.current_bidder_id = None
            self.sold = False

        def reset(self):
            self.current_bid = 0
            self.current_bidder_id = None
            self.sold = False
            self.stats = np.random.rand(3)

    class Bidder:
        """
        Represents a bidder in an English auction.

        Attributes:
            budget (int): The budget of the bidder.
            spending (int): The spending of the bidder.
        """
        def __init__(self, id : int, budget: int) -> None:
            self.id = id
            self.budget = budget
            self.spending = 0
            self.won_objects = []
            self.value_won_so_far = 0
            # weights for the stats of the object
            self.prefs = np.random.rand(3)

        def reset(self):
            self.spending = 0
            self.won_objects = []
            self.value_won_so_far = 0
            self.prefs = np.random.rand(3)

    def __init__(
        self,
        num_agents: int = 2,
        render_mode: str = None,
        bid_increments: int = 1,
        budget_limit: int = 6,
        num_objects: int = 3,
        no_bid_rounds: int = 1,
    ) -> None:
        """
        Initializes an instance of the EnglishAuctionEnv class.

        Args:
            num_agents (int): The number of agents participating in the auction. Defaults to 2.
            render_mode (str): The rendering mode for visualization. Defaults to None.
            bid_increments (int): The increment value for each bid. Defaults to 1.
            budget_limit (int): The budget limit for each agent. Defaults to 6.
            num_objects (int): The number of objects available for auction. Defaults to 3.
            no_bid_rounds (int): The number of rounds for which an object can remain unsold. Defaults to 1.

        Currently all the values are initialized randomly
        """
        super().__init__()

        # auction parameters
        self.num_agents = num_agents
        self.bid_increment = bid_increments
        self.budget_limit = budget_limit
        self.num_objects = num_objects

        # number of rounds for which an object can remain unsold
        self.no_bid_rounds = no_bid_rounds
        self.no_bid_counter = 0

        # initialize objects
        self.objects = [self.BiddingObject(i) for i in range(self.num_objects)]
        self.current_object = self.objects[0]

        self.bid_history_size = 5

        # initialize agents
        self.agents = [self.Bidder(i, self.budget_limit) for i in range(self.num_agents)]
        
        # The observations would be the following features:
        # 1. One-hot encoded current object (env.num_objects)
        # 2. All object stats (3 * env.num_objects)
        # 3. Current bidding price (1)
        # 4. Bid increment (1)
        # 5. Number of no bid rounds right now (1)
        # 6. Flattened action history (env.num_agents * env.bid_history_size)
        self.observation_space = spaces.Dict({
            "object_id": spaces.Discrete(self.num_objects),
            "stats": spaces.Box(low=0, high=1, shape=(3 * self.num_objects,)),
            "current_bid": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "bid_increment": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "no_bid_rounds": spaces.Discrete(self.no_bid_rounds),
            "action_history": spaces.Box(low=0, high=1, shape=(self.num_agents * self.bid_history_size,))
        })

        # At each time step, each agent can either bid or pass. 
        # Action space will be a vector of size num_agents with each element representing the action of an agent
        self.action_space = spaces.MultiDiscrete([2] * self.num_agents)
        self.render_mode = render_mode

        # rewards buffer that is updated at each step
        self.rewards = np.zeros(self.num_agents)

        # action history queue
        self.action_history = np.zeros((self.num_agents, self.bid_history_size)) -1
        


    def __get_observation(self):
        """
        Returns the current observation of the environment.

        Returns:
            dict: The observation of the environment.
        """
        # One-hot encode the current object
        object_id = np.zeros(self.num_objects)
        object_id[self.current_object.id] = 1

        # Flatten the stats of all objects
        stats = np.concatenate([obj.stats for obj in self.objects])

        # Get the current bid
        current_bid = np.array([self.current_object.current_bid if self.current_object.current_bid else 0])

        # Get the bid increment
        bid_increment = np.array([self.bid_increment])

        # Get the number of no bid rounds
        no_bid_rounds = np.array([self.no_bid_counter])

        # Get the action history
        action_history = self.action_history.flatten()

        return {
            "object_id": object_id,
            "stats": stats,
            "current_bid": current_bid,
            "bid_increment": bid_increment,
            "no_bid_rounds": no_bid_rounds,
            "action_history": action_history,
        }
    
    def __get_info(self):
        """
        Returns the agent preferences, the object winners so far and their remaining budgets.

        Returns:
            dict: The auction properties
        """
        return {
            "agent_prefs": [agent.prefs for agent in self.agents],
            "agent_objects_won": [agent.won_objects for agent in self.agents],
            "remaining_budgets": [agent.budget - agent.spending for agent in self.agents],
        }

    def reset(self, *, seed = None, options: None = None):
        super().reset(seed=seed, options=options)
        # reset agents
        for agent in self.agents:
            agent.reset()
        # reset object stats
        for obj in self.objects:
            obj.reset()
        # reset current object
        self.current_object = self.objects[0]
        return self.__get_observation(), self.__get_info()
    
    def step(self, action):
        # action is a vector of size num_agents with each element representing the action of an agent
        # 0 - pass, 1 - bid
        terminated = False
        self.rewards = np.zeros(self.num_agents)

        # add action to action history
        self.action_history = np.roll(self.action_history, 1, axis=1)
        self.action_history[:, 0] = action
        
        bidders = []
        # if even a single agent bids, the current bid is incremented by bid_increment
        for i, bid in enumerate(action):
            if bid == 1:
                if self.agents[i].budget - self.agents[i].spending >= self.current_object.current_bid + self.bid_increment:
                    bidders.append(i)
                else:
                    print(f"Agent {i} does not have enough budget to bid.")
                    # reward penalize the agent for trying to bid without enough budget
                    self.rewards[i] = -5

        if bidders:
            chosen_bidder = np.random.choice(bidders)
            if len(bidders) > 1:
                print(f"Picked bidder {chosen_bidder} randomly from {bidders}")
            self.current_object.current_bid += self.bid_increment
            self.current_object.current_bidder_id = chosen_bidder
        else:
            self.no_bid_counter += 1
        
        # if no_bid_counter exceeds no_bid_rounds, the object is considered sold to the current_bidder
        if self.no_bid_counter >= self.no_bid_rounds:
            # if there is no current_bidder, the object remains unsold
            if self.current_object.current_bidder_id is not None:
                self.current_object.sold = True
                self.agents[self.current_object.current_bidder_id].spending += self.current_object.current_bid
                self.agents[self.current_object.current_bidder_id].won_objects.append(self.current_object.id)
                # reward the agent for winning the object
                self.rewards[self.current_object.current_bidder_id] = 1
                
                # here rewards are only an indication of winning. We will have a reward transformation in the agent class to make it more meaningful
                self.rewards[self.current_object.current_bidder_id] = 1
                # indicate that other agents did not win
                for i in range(self.num_agents):
                    if i != self.current_object.current_bidder_id:
                        self.rewards[i] = -1
            self.no_bid_counter = 0
            # move to the next object
            if self.current_object.id == self.num_objects - 1:
                # all objects done, conclude the auction
                terminated = True
            else:
                self.current_object = self.objects[(self.current_object.id + 1) % self.num_objects]
        
        # we can terminate if all the agents have exhausted their budgets
        if all([agent.budget - agent.spending <= 0 for agent in self.agents]):
            terminated = True

        return self.__get_observation(), self.rewards, terminated, False, self.__get_info()