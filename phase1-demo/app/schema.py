from pydantic import BaseModel, Field
from typing import List, Optional

class Room(BaseModel):
    id: str = Field(..., description="Unique identifier for the room (e.g., living_room_1)")
    type: str = Field(..., description="Type of the room (e.g., living_room, bedroom, kitchen, bathroom)")
    count: int = Field(default=1, description="Number of such rooms")
    size: Optional[str] = Field(None, description="Size description (e.g., large, small)")
    connections: List[str] = Field(default_factory=list, description="IDs or types of rooms this should connect to")

class Requirements(BaseModel):
    bedrooms: int = Field(default=0, description="Total required bedrooms")
    bathrooms: int = Field(default=0, description="Total required bathrooms")
    kitchen: int = Field(default=0, description="Total required kitchens")
    living_room: int = Field(default=0, description="Total required living rooms")

class ApartmentState(BaseModel):
    rooms: List[Room] = Field(default_factory=list, description="List of specifically identified rooms")
    requirements: Requirements = Field(default_factory=Requirements, description="Overall numeric requirements for the apartment")
