import torch

class RK4_Solver:
    def step(self, x, f, dt, *args):
        """
        Performs one step of RK4 integration
        
        Args:
            x: Current state
            f: Function that computes derivatives (dx/dt = f(x, t, *args))
            *args: Additional arguments to pass to f
            
        Returns:
            Next state after dt
        """
        
        # RK4 steps
        k1 = f(x, *args)
        k2 = f(x + dt/2 * k1, *args)
        k3 = f(x + dt/2 * k2, *args)
        k4 = f(x + dt * k3, *args)
        
        # Update state
        return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # return x + dt * k2
        # return x + dt * k1 