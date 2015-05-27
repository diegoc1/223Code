# Simulates the gambling game of Lecture # 1
#
# MS&E 223: Simulation
#
# Usage: python gamble.py -n k [-t -i -d m]
# where -n (--num) k: Number of replications given by k
# -t (--trial): Trial run to get required number of replications
# -i (--confint): Calculate the CI of the point estimator
# -d (--debug) m: Verbose output needed, levels from 0-2
#
# Uses the clcg4 module (the module should be in the python path or the same folder)
__author__ = "Neeraj Pradhan"

import clcg4
from math import sqrt, log, exp, cos, sin, pi
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import random
import math

class Estimator:
    """ Computes point estimates and confidence intervals """
    def __init__(self, z, conf_str):
        self.k = 0  # number of values processed so far
        self.sum = 0.0  # running sum of values
        self.v = 0.0  # running value of (k-1)*variance
        self.z = float(z) # quantile for normal Confidence Interval
        self.conf_str = conf_str # string of form "xx%" for xx% Confidence Interval
        self.list_of_values = []

    def reset(self):
        self.k = 0
        self.sum = 0
        self.v = 0

    def process_next_val(self, value):
        self.list_of_values.append(value)
        self.k += 1
        if self.k > 1:
            diff = self.sum - (self.k - 1) * value
            self.v += diff/self.k * diff/(self.k-1)
        self.sum += value

    def get_variance(self):
        if self.k > 1:
            var = self.v/(self.k-1)
        else:
            raise RuntimeError("Variance undefined for number of observations = 1")
        return var

    def get_mean(self):
        return self.sum/self.k if self.k > 1 else 0

    def get_conf_interval(self):
        hw = self.z * sqrt(self.get_variance()/self.k)
        point_est = self.get_mean()
        c_low = point_est - hw
        c_high = point_est + hw
        return self.conf_str + " Confidence Interval [ %.4f" %c_low +  ", %.4f" %c_high + "]"

    def get_num_trials(self, epsilon, relative=True):
        var = self.get_variance()
        width = self.get_mean() * epsilon if relative else epsilon
        return int((var * self.z * self.z)/(width * width))







# class MarkovChainHolder:
#     def __init__(self, k, n):


import heapq  
 
class EventList:
    """Event list using a heap data structure"""
    def __init__(self):
        self.event_list = []
        self.event_record = {}

    def add_event(self, event_id, time):
        """Adds event_id with time in the event list"""
        if event_id in self.event_record:
            self.cancel_event(event_id)
        new_event = [time, event_id]
        self.event_record[event_id] = new_event
        heapq.heappush(self.event_list, new_event)

    def cancel_event(self, event_id):
        """Cancels event having id = event_id"""
        if event_id in self.event_record:
            to_remove = self.event_record.pop(event_id)
            to_remove[-1] = "<canceled>"
        else:
            raise KeyError("Event %s not in list." %str(event_id))

    def next_event(self):
        """Return tuple containing (event_id, time)"""
        if self.event_list:
            next_event = heapq.heappop(self.event_list)
            if next_event[-1] != "<canceled>":
                del self.event_record[next_event[-1]]
                return next_event[-1], next_event[0]
            return self.next_event()
        else:
            raise KeyError("Popping from an empty event list.")

    def reset(self):
        self.event_list = []
        self.event_record = {}

    def __str__(self):
        el = sorted(self.event_list)
        return "\n".join([str(x) for x in el])



class AmbulanceSimulator:
    def __init__(self, unigen, number_of_ambulances=3, U=10, beta=0.5):
        self.unigen = unigen
        self.upper_bound = U
        self.beta = beta
        self.previous_time = 0
        self.number_non_dispatched = 0
        self.number_of_ambulances = number_of_ambulances
        self.ambulance_states = [0] * self.number_of_ambulances
        self.running_count_of_idle_ambs = 0
        self.running_count_of_waiting_time = 0

        self.regeneration_times = []
        self.regeneration_idle_ambulances = []
        self.previous_cycle_ending_time = 0

        self.event_list = EventList()
        self.initialize_events()



    def initialize_events(self):
        #Add first arrival event
        first_arrival_event = (0,0)
        time_till_event = self.sample_from_poisson_process()
        self.event_list.add_event(first_arrival_event, time_till_event)

    def sample_from_log_logistic(self, alpha=3.0, gamma=.15):
        u = self.unigen.next_value(1)
        log_val = gamma * pow(((1.0 / u) - 1.0), (-1.0 / alpha))
        return log_val


    def sample_from_poisson_process(self):
        u = self.unigen.next_value(2)
        lambda_val = 1.0 / self.beta
        log_val = -1 * math.log(u) / (1.0 * lambda_val)
        return log_val

    #Get ambulance state
    def get_state_of_ambulance(self, amb_id):
        return self.ambulance_states[amb_id - 1]

    #Set ambulance state
    def set_state_of_ambulance(self, amb_id, new_state):
        self.ambulance_states[amb_id - 1] = new_state

    def get_random_ambulance(self):
        amb_choices = []
        for amb_id in range(1, 4):
            #Check if ambulance is idle
            if self.get_state_of_ambulance(amb_id) == 0:
                amb_choices.append(amb_id)

        if len(amb_choices) > 0:
            return random.choice(amb_choices)
        else:
            return -1

    def get_number_of_idle_ambulances(self):
        # print "States", self.ambulance_states
        free_ambs = sum([1 if self.get_state_of_ambulance(amb_id) == 0 else 0 for amb_id in range(1, 4)])
        return free_ambs

    def compute_taylor_series_confidence_interval(self):
        return

    def run_simulation(self, debug=False, regeneration=True):

        count = 0
        first_iter = True
        while(True):
            if first_iter:
                first_iter = False
            

            #Break after 100 hours
            if self.previous_time > 100 and not regeneration:
                break

            if debug:
                print "Current events"
                print "==============================="
                print
                print self.event_list

            #Get next event           
            event, curr_time = self.event_list.next_event()
            amb_id, event_type = event

            if debug:
                print "Triggered event", event
                print
                print "==============================="
                import pdb;pdb.set_trace()

            #############################################################
            #################### Part 1b  START #########################
            #############################################################
            if regeneration:
                # print self.ambulance_states
                # import pdb;pdb.set_trace()
                if sum(self.ambulance_states) == 0 and self.number_non_dispatched == 0 and not first_iter:

                    #Update cycle times
                    new_cycle_time = curr_time - self.previous_cycle_ending_time
                    self.previous_cycle_ending_time = curr_time
                    self.regeneration_times.append(new_cycle_time)

                    #Update idle counts
                    self.running_count_of_idle_ambs /= new_cycle_time
                    self.regeneration_idle_ambulances.append(self.running_count_of_idle_ambs)

                    self.running_count_of_idle_ambs = 0

                    if len(self.regeneration_times) == 500:
                        break
            #############################################################
            ##################### Part 1b END ###########################
            #############################################################


            #Update running count
            holding_time = curr_time - self.previous_time
            self.previous_time = curr_time
            self.running_count_of_idle_ambs += holding_time * self.get_number_of_idle_ambulances()

            #Update patient waiting time count
            if self.number_non_dispatched > 0:
                self.running_count_of_waiting_time += holding_time

            #If new arrival is triggered
            if event_type == 0:

                #Schedule new arrival if limit is not exceeded
                if self.number_non_dispatched <= self.upper_bound:
                    new_arrival = (0, 0)
                    time_for_arrival = self.sample_from_poisson_process()
                    updated_time = curr_time + time_for_arrival
                    self.event_list.add_event(new_arrival, updated_time)

                #Find ambulance to service new arrival
                amb_id = self.get_random_ambulance()

                if amb_id == -1: #If no ambulance can service new arrival, increment self.number_non_dispatched and don't update any states
                    self.number_non_dispatched += 1
                    continue

            ###############################################################
            ######ONLY REACHED IF AMBULANCE STATE NEEDS TO BE UPDATED######
            ###############################################################

            #Update ambulance state
            curr_amb_state = self.get_state_of_ambulance(amb_id)
            new_amb_state = (curr_amb_state + 1) % 4


            #Updated event
            new_event_type = (event_type + 1) % 4

            #If ambulance has returned, check to see if there are any non-dispatched people
            if new_amb_state == 0:

                #If there is a non dispatched person, then immediately dispatch the ambulance
                if self.number_non_dispatched > 0:
                    new_amb_state = 1
                    new_event_type = 1

                    self.number_non_dispatched -= 1   

                    #Schedule new arrival if num people waiting is brought below the limit
                    if self.number_non_dispatched == (self.upper_bound - 1):
                        new_arrival = (0, 0)
                        time_for_arrival = self.sample_from_poisson_process()
                        updated_time = curr_time + time_for_arrival
                        self.event_list.add_event(new_arrival, updated_time)

                    


            self.set_state_of_ambulance(amb_id, new_amb_state)


            #If ambulance is not idle, create a new ambulance event
            if new_event_type != 0:            
                new_event = (amb_id, new_event_type)

                #Get proper gamma
                g = .15
                if amb_id == 3:
                    g = .3 
                time_for_new_event = self.sample_from_log_logistic(gamma=g)
                updated_time = curr_time + time_for_new_event
                self.event_list.add_event(new_event, updated_time)

        ####################################################
        ################ FINAL CALCULATIONS ################
        ####################################################
        self.running_count_of_idle_ambs /= (curr_time * 1.0)
        print "Running total", self.running_count_of_idle_ambs

        self.running_count_of_waiting_time /= (curr_time * 1.0)
        print "Waiting time fraction", self.running_count_of_waiting_time

        if regeneration:
            mean_cycle_time = sum(self.regeneration_times) / len(self.regeneration_times)
            mean_num_idle = sum(self.regeneration_idle_ambulances) / len(self.regeneration_idle_ambulances)
        print "Mean cycle time", mean_cycle_time
        print "Mean num idle", mean_num_idle

        if regeneration:
            self.compute_taylor_series_confidence_interval(self.regeneration_times, self.regeneration_idle_ambulances)
            return mean_num_idle / mean_cycle_time




        return self.running_count_of_waiting_time




def do_rep(unigen, est):
    curr_g1q = AmbulanceSimulator(unigen)
    val = curr_g1q.run_simulation()
    #print "val", val
    #import pdb; pdb.set_trace()

    est.process_next_val(val)
    return val

if __name__ == "__main__":
    # parse command line arguments
    parser = ArgumentParser(description = "gamble -n [--trial --confint --debug m]")
    parser.add_argument('-n', '--num', help="Number of replications", required=True)
    parser.add_argument('-t', '--trial', help="Trial run to get required number of replications", action='store_true')
    parser.add_argument('-i', '--confint', help="Calculate the CI of the point estimator", action='store_true')
    parser.add_argument('-d', '--debug', help="Verbose output needed", default=0)
    sysargs = parser.parse_args()

    est = Estimator(1.96, "95%")  # 95% CI
    epsilon = 0.005  # Determines the width of the CI
    unigen = clcg4.Clcg4()  # Instantiate the random number generator
    unigen.init_default()

    # verbose printing for different debug levels
    def verbose_print(level, *args):
        if level <= int(sysargs.debug):
            for arg in args:
                print arg,
        else:
            pass

    # reinitialize generator for production runs
    if not sysargs.trial: unigen.init_generator(1, clcg4.NEW_SEED)

    list_of_values = []
    # run simulation repetitions and collect stats
    for rep in range(int(sysargs.num)):
        print "rep", rep
        val = do_rep(unigen, est)
        list_of_values.append(val)
        verbose_print(1, "Repetition", rep+1, " : %.2f" %val, "\n\n")


    def perform_sectioning(num_sections=5, num_observations=100, inner_quartile=.25, outer_quartile=.75):

        section_estimator = Estimator(1.96, "95%")

        for curr_section in range(num_sections):
            next_section = curr_section + 1
            curr_section = list_of_values[curr_section * num_observations:next_section * num_observations]
            curr_section = sorted(curr_section)
            inner_idx = int(math.ceil(inner_quartile * num_observations))
            outer_idx = int(math.ceil(outer_quartile * num_observations))
            range_length = curr_section[outer_idx] - curr_section[inner_idx]
            print "Range: " + str(curr_section[inner_idx]) + "   -   " + str(curr_section[outer_idx])
            section_estimator.process_next_val(range_length) 

        print "Variance", section_estimator.get_variance()
        print "Mean", section_estimator.get_mean()

        std_dev = math.sqrt(section_estimator.get_variance())
        mean = section_estimator.get_mean()
        t = 2.132

        lower_conf_intvl = mean - (std_dev * t / math.sqrt(5))
        upper_conf_intvl = mean + (std_dev * t / math.sqrt(5))

        print "lower_conf_intvl", lower_conf_intvl
        print "upper_conf_intvl", upper_conf_intvl


    perform_sectioning()

    # print results
    verbose_print(0, "Average net gain: %.3f" %est.get_mean())
    if sysargs.confint:
        verbose_print(0, "with", est.get_conf_interval())
    print "\n"
    if sysargs.trial:
        epsilon = .05 * est.get_mean()
        print "Epsilon", epsilon
        print est.get_conf_interval()
        print "Est. # of repetitions for +/-", epsilon, "accuracy: ", est.get_num_trials(epsilon, False)





