ISMRM_RRSG
===================================

SHORT DESCRIPTION
-------------------------

Purpose of this piece of software:
    
    Supply a (new) user to MRI with software that allows easy reproducibility of famous papers in the field of MRI
    
The means of attaining this

    Provide generic software that allows for the reproduction of several reconstruction techniques.
    
The (higher) goal of this project

    Work towards a community where work is easiliy verified

    
Programming questions/decision
-------------------------

- Do we need to buid GPU support?
    Or is CPU support good enough?
    
- Do we want it to be a full python implementation?
    Or do we allow other extension as well? (c/cpp/matlab)
    
- Do we want it to be dependant on as little packages as possible?
    Or do we want to re-use existing python packages? (medutils, pynufft, our own packages, ...)
    
- Do we want to have it heavily commented, almost tutorial like?
    Or can we assume most steps how they are done and why?
    
- Do we want to offer a jupyter-notebook as "final result"?
    Or rather a package that people can use?    
    
- Do we want to offer one single technique for solving this problem?
    Or do we want to add multiple ways? (think of optimization, density compensation, itensity approximation, ...)



Quick Installing Guide:
-------------------------


How to run the reconstruction:
-------------------------


Citation/References:
-------------------------

