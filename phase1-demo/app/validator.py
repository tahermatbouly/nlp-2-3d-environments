from .schema import ApartmentState

def is_state_complete(state: ApartmentState) -> bool:
    reqs = state.requirements
    
    # We consider the state complete if at least the basic requirements are specified,
    # meaning the user has explicitly requested at least one bedroom, one bathroom,
    # a kitchen, and a living room.
    
    return (
        reqs.bedrooms >= 1 and 
        reqs.bathrooms >= 1 and 
        reqs.kitchen >= 1 and 
        reqs.living_room >= 1
    )
