import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1
        
        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([10.,10.,10.]) 

    def get_reward(self,done):
        """Uses current pose of sim to return reward."""
        x, y, z = self.sim.pose[0:3]
        x_a, y_a, z_a = self.sim.pose[3:6]
        xdot, ydot, zdot = self.sim.v[0:3]
        xdot_a, ydot_a, zdot_a = self.sim.angular_v[0:3]
        time = self.sim.time
        target_z = self.sim.pose[2]

        reward = 1.-.3*(abs(self.sim.pose[:3]-self.target_pos[:3])).sum()
        
#        reward = 1.
        
#        reward = .1-.03*abs(self.sim.pose[:3]-#self.target_pos[:3]).sum()-.01*abs(ydot_a)-.01*abs(zdot_a)
            
#        if time > self.runtime:
#            reward += 1.
        
#        if z > target_z and z <(target_z + 10):
#            reward += 10.
            
#        if done and time < self.runtime:
#            reward += - 1 / time
        
        return reward  

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done) 
           
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state