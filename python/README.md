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
    - In my opinion CPU support is good enough. It should be sort of an educational example. GPU's can be added later on if need be.
    
- Do we want it to be a full python implementation?
    Or do we allow other extension as well? (c/cpp/matlab)
    - I would prefer a full python implementation to allow for as easy as possible usage/understanding
    
- Do we want it to be dependant on as little packages as possible?
    Or do we want to re-use existing python packages? (medutils, pynufft, our own packages, ...)
    - As little as possible because of the same reasons as above.
    
- Do we want to have it heavily commented, almost tutorial like?
    Or can we assume most steps how they are done and why?
    - Heavily commented
    
- Do we want to offer a jupyter-notebook as "final result"?
    Or rather a package that people can use?    
    - A package would be my suggestions. The repo is set up to produce one. This would allow for pip installation from PyPI for example. However, if someone realy wants a notebook I am fine with that too.
    
- Do we want to offer one single technique for solving this problem?
    Or do we want to add multiple ways? (think of optimization, density compensation, itensity approximation, ...)
    - I would suggest to start with a single one.



Quick Installing Guide:
-------------------------


How to run the reconstruction:
-------------------------


Citation/References:
-------------------------

