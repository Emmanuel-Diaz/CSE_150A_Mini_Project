#Import dependencies
import numpy as np
from collections import defaultdict

#Initialize global variables
name = 0
criteria = 0
beer_freq = 0
beer_criteria = 0
epsilon = 0
number_of_episodes = 0
num_iterations = 0

#Loads in the beer dataset
def load_data():   
    beer_file = "beer_50000.json"
    def parseReviews(fname):
        for l in open(fname):
            yield eval(l)
    return list(parseReviews(beer_file)) 

#Gets the beer frequencies and the summed criteria
def get_freq_and_criteria(crit):
    beer_crit = defaultdict(float)
    beer_freq = defaultdict(int)
    
    if crit in ["appearance","palate","taste","overall"]:
        crit = "review/"+crit
    else:
        crit = "beer/"+crit.upper()
    
    for beer in data:
        curr_beer = beer['beer/style'] 
        beer_freq[curr_beer] += 1
        beer_crit[curr_beer] += beer[crit]
    return beer_freq,beer_crit


#Gets the prior probabilties for given beer criteria
def compute_prior_probs(beer_crit):
    
    #Compute priors
    prior_probs = [beer_crit[style]/total for style,total in beer_freq.items()]
    max_prob = max(prior_probs)
    min_prob = min(prior_probs)
    
    #Normalize probabilities
    prior_probs = [((p-min_prob)/(max_prob-min_prob)) for p in prior_probs]
    new_max_prob = max(prior_probs)
    new_min_prob = min(prior_probs)
    
    new_probs = []
    for p in prior_probs:
        
        #Add/Subtract noise to max/min probabilities
        if p == new_max_prob:
            new_probs.append(p-float(np.random.uniform(0.005,0.1,1)))
        elif p == new_min_prob:
            new_probs.append(p+float(np.random.uniform(0.005,0.1,1)))
        else:
            new_probs.append(p)
    return new_probs
       
        
class Environment:

    def __init__(self, probs):
        self.probs = probs  # success probabilities for each arm

    def step(self, action):
        return 1 if (np.random.random()  < self.probs[action]) else 0


class Agent:

    def __init__(self, num_styles_beers, eps):
        self.nActions = num_styles_beers #Number of types of beers
        self.eps = epsilon # Epsilon
        self.n = np.zeros(nActions, dtype=np.int) # Action counts
        self.Q = np.zeros(nActions, dtype=np.float) # Q value

    def update_Q(self, action, reward):
        # Update Q action-value given (action, reward)
        self.n[action] += 1
        self.Q[action] += (1.0/self.n[action]) * (reward - self.Q[action])

    # Epsilon-greedy policy
    def get_action(self):
  
        #Exploration
        if np.random.random() < self.eps:
            return np.random.randint(self.nActions)
        
        #Exploitation
        else:
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))

        
# Start multi-armed bandit simulation
def multi_beer_fanatic(probs, N_episodes):
    env = Environment(probs) # Start Environment
    agent = Agent(len(probs), epsilon)  # Start Agent
    actions, rewards = [], []
    for episode in range(N_episodes):
        action = agent.get_action() # Get random policy
            reward = env.step(action) # Take step and get reward
        agent.update_Q(action, reward) # update Q value
        actions.append(action)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)


def nights_out_trying_beer(beers_tried,nights_out):
    # Run multi-armed bandit experiments
    print("Running Multi-Beer Fanatic with num_styles_beers = {}, epsilon = {}".format(len(probs), epsilon))
    
    R = np.zeros((beers_tried,))  # Reward history
    A = np.zeros((beers_tried, len(probs)))  # Action history
    
    for i in range(nights_out):
        actions, rewards = multi_beer_fanatic(probs, beers_tried)  # Run Beer Fanatic
        if (i + 1) % (nights_out / 100) == 0:
            print("[Night {}/{}] ".format(i + 1, nights_out) +
                  "num_beers = {}, ".format(beers_tried) +
                  "Average Reward = {}".format(np.sum(rewards) / len(rewards)))
        R += rewards
        
        #Increment actions
        for j, a in enumerate(actions):
            A[j][a] += 1 
            
    return A


#Find the beer with the best action
def find_beer(ep, num_epi, num_iter):
    A = nights_out_trying_beer(num_epi,num_iter)
    max_ind = 0
    max_val = 0
    for i in range(len(probs)):
        action_vals = 100 * A[:,i] / num_iter
        curr_val = action_vals[-1]
        if curr_val > max_val:
            max_val = curr_val
            max_ind = i
    return list(beer_freq.keys())[max_ind]



#Executes console prompts for user input of parameters for model
def console_prompts():      
    print("What are you looking for in a beer? Is it:\n How it looks? (Appearance)\n How it goes down? (Palate)\n How it rests in your mouth? (Taste)\n How much you wanna get wasted? (ABV)\n Just give me your best beer (overall)\n")
    global criteria
    criteria = input()
    while criteria.lower() not in ["appearance","palate","taste","abv","overall"]:
        print("Not one that I asked, please select one of the following")
        criteria = input("Appearance\nPalate\nTaste\nABV\nOverall\n")
    
    global beer_freq
    global beer_criteria
    beer_freq,beer_criteria= get_freq_and_criteria(criteria)
    
    print("On a scale from 0 to 10, how adventurous are you to try new beers?\n")
    adventure_level = input()
    while not adventure_level.isdecimal() or float(adventure_level)>10.0 or float(adventure_level)<0.0:
        print("Not a valid number, enter an integer between 0 and 10")
        adventure_level = input()
    
    global probs
    probs = compute_prior_probs(beer_criteria)
    global epsilon 
    epsilon = float(adventure_level)/10
    
    print("How many beers would you like to try? (Number of episodes)")
    num_beers = input()
    while not num_beers.isdecimal() or float(num_beers)>50000 or float(num_beers)<1:
        print("Not a valid number, enter an integer between 0 and 50000")
        num_beers = input()
        
    global number_of_episodes 
    number_of_episodes = int(num_beers)
        
    print("How many nights would you like to go out drinking?")
    num_nights_out = input()
    while not num_nights_out.isdecimal() or float(num_nights_out)>50000 or float(num_nights_out)<1:
        print("Not a valid number, enter an integer between 1 and 50000")
        num_nights_out = input()
               
    global num_iterations 
    num_iterations = int(num_nights_out)

    
#----------- MAIN PROGRAM -----------#
if __name__=="__main__":
    print("Welcome to the Multi-Beer Fanatic! Get ready to try your next beer, but first a few questions...")
    
    #Load data
    data = load_data()
    name = input("What should I call you?  ")
    print("Hello {}, I need to ask you a bit about your beer experience.".format(name)) 
    console_prompts()
        
    print("Finding you the right beer...")
    while(1):
        exit_program = False
        reccommended_beer = find_beer(epsilon,number_of_episodes,num_iterations)
        print("{}, after all that drinking, I really recommend".format(name), reccommended_beer)
        print("Would you like to try this simulation again, change your question answers or exit?")
        ans = input("[Again/Change/Exit]  ")
        if ans.lower() == "again":
            continue
        elif ans.lower() == "change":
            print("Please enter your new answers to the questions:")
            console_prompts()
        else:
            print("Thank you for using the Multi-Beer Fanatic, sit back, relax and enjoy that recommended {}".format(recommended_beer))
            exit_program = True
            break
        while ans.lower() not in ["again","change","exit"]:
            print("Please choose one of the following options: [Again/Change]")
            ans = input()
            if ans.lower() == "again":
                break
            elif ans.lower() == "change":
                print("Please enter your new answers to the questions:")
                console_prompts()
                break
            else:
                print("Thank you for using the Multi-Beer Fanatic, sit back, relax and enjoy that recommended {}".format(reccommended_beer))
                exit_program = True
                break
        if exit_program:
            break
        
    