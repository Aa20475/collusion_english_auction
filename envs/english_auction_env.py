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
            self.current_bid = None
            self.current_bidder_id = None
            self.sold = False


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

        # keep track of current spending of each agent to enforce budget limit
        self.agent_spending = np.zeros(self.num_agents)

        # the observation space will be the current object being auctioned
        self.observation_space = spaces.Dict(
            {
                "object_id": spaces.Discrete(self.num_objects),
                "stats": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                "current_bid": spaces.Discrete(self.budget_limit),
                "current_bidder_id": spaces.Discrete(self.num_agents),
            }
        )

        # At each time step, each agent can either bid or pass. 
        # Action space will be a vector of size num_agents with each element representing the action of an agent
        self.action_space = spaces.MultiDiscrete([2] * self.num_agents)
        self.render_mode = render_mode

        # rewards buffer that is updated at each step
        self.rewards = np.zeros(self.num_agents)

    def __get_observation(self):
        """
        Returns the current observation of the environment.

        Returns:
            dict: The observation of the environment.
        """
        return {
            "object_id": self.current_object.id,
            "stats": self.current_object.stats,
            "current_bid": self.current_object.current_bid,
            "current_bidder_id": self.current_object.current_bidder_id,
        }
    
    def __get_info(self):
        """
        Returns the auction properties

        Returns:
            dict: The auction properties
        """
        return {
            "num_agents": self.num_agents,
            "bid_increment": self.bid_increment,
            "budget_limit": self.budget_limit,
            "num_objects": self.num_objects,
            "no_bid_rounds": self.no_bid_rounds,
        }

    def reset(self, *, seed: int | None = None, options: None):
        super().reset(seed=seed, options=options)
        # reset agent spending
        self.agent_spending = np.zeros(self.num_agents)
        # reset object stats
        for obj in self.objects:
            obj.stats = np.random.rand(3)
            obj.current_bid = None
            obj.current_bidder_id = None
            obj.sold = False
        # reset current object
        self.current_object = self.objects[0]
        return self.__get_observation(), self.__get_info()
    
    def step(self, action):
        # action is a vector of size num_agents with each element representing the action of an agent
        # 0 - pass, 1 - bid
        terminated = False
        self.rewards = np.zeros(self.num_agents)
        
        bid = False
        # if even a single agent bids, the current bid is incremented by bid_increment
        for i, bid in enumerate(action):
            if bid == 1:
                self.current_object.current_bid += self.bid_increment
                self.current_object.current_bidder_id = i
                bid = True
        
        # if no agent bids, the no_bid_counter is incremented
        if not bid:
            self.no_bid_counter += 1
        
        
        # if no_bid_counter exceeds no_bid_rounds, the object is considered sold to the current_bidder
        if self.no_bid_counter >= self.no_bid_rounds:
            # if there is no current_bidder, the object remains unsold
            if self.current_object.current_bidder_id is not None:
                self.current_object.sold = True
                self.agent_spending[self.current_object.current_bidder_id] += self.current_object.current_bid
                # here rewards are only an indication of winning. We will have a reward transformation in the agent class to make it more meaningful
                self.rewards[self.current_object.current_bidder_id] = 1
                # indicate that other agents did not win
                for i in range(self.num_agents):
                    if i != self.current_object.current_bidder_id:
                        self.rewards[i] = -1
            self.no_bid_counter = 0
            # move to the next object
            if self.current_object.id == self.num_objects - 1:
                # conclude the auction
                terminated = True
            else:
                self.current_object = self.objects[(self.current_object.id + 1) % self.num_objects]
        
        return self.__get_observation(), self.rewards, terminated, False, self.__get_info()