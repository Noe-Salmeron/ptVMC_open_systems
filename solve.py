from ptVMC_no_sampling import ptVMC_exact, ptVMC_implicit_exact
from ptVMC_sampling import ptVMC, ptVMC_factored_2nd, ptVMC_implicit, ptVMC_factored_3rd

###

# A centralized function to run any version of the algoritm by simply changing one keyword.
# Note that the implicit samplers are not supposed to work since doing a timestep backward in time is
# ill defined for a dissipative system.

###


def solve(lind, vstate, stop_time, dt, method, **kwargs):
    # solves the dynamics with the choosen method. The model is defined by lind and the initial state by vstate.
    # TODO For now all the methods return the result under a different format, they should write their results in a file with the same format

    if method == "Exact":
        from exact_solver import exact_solver # imports are here to avoid having to install qutip if the exact solver is not used

        return exact_solver(lind, vstate, stop_time, dt, **kwargs)
    
    elif method == "ptVMC":
        return ptVMC(lind, vstate, stop_time, dt, **kwargs)
    elif method == "ptVMC_implicit":
        return ptVMC_implicit(lind, vstate, stop_time, dt, **kwargs)
    
    elif method == "ptVMC_no_sampling":
        return ptVMC_exact(lind, vstate.model, {"params":vstate.parameters}, stop_time, dt, **kwargs)
    elif method == "ptVMC_no_sampling_implicit":
        return ptVMC_implicit_exact(lind, vstate.model, {"params":vstate.parameters}, stop_time, dt, **kwargs)
    
    if method == "ptVMC_factored_2":
        return ptVMC_factored_2nd(lind, vstate, stop_time, dt, **kwargs)
    if method == "ptVMC_factored_3":
        return ptVMC_factored_3rd(lind, vstate, stop_time, dt, **kwargs)
    
    else:
        raise ValueError("The choosen method is invalid")