from .schema import ApartmentState

# In-memory store mapping session_id to state
conversation_states = {}

def get_state(session_id: str = "default") -> ApartmentState:
    if session_id not in conversation_states:
        conversation_states[session_id] = ApartmentState()
    return conversation_states[session_id]

def update_state(session_id: str, new_state: ApartmentState):
    conversation_states[session_id] = new_state

def reset_state(session_id: str = "default"):
    conversation_states[session_id] = ApartmentState()
