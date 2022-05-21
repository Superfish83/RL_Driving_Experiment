import DriveSimulator as dsim
import random

if __name__ == '__main__':
    sim = dsim.DriveSimulator()
    while True:
        sim.reset()
        sim_over = False
        s_t = sim.get_sim_state()
        
        while not sim_over:
            a_t = sim.agent.decide_action(s_t)
            s_tp1, r_t, sim_over = sim.step(a_t)
            sim.agent.process_step(s_t, a_t, r_t, s_tp1, sim_over)
            
            s_t = s_tp1

        sim.agent.train()
