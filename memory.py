from typing import List
import datetime

class Memory:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories: List[str] = []

    def add_memory(self, memory: str) -> None:
        timestamp = datetime.datetime.now().isoformat()
        self.memories.append(f"{timestamp}: {memory}")
        if len(self.memories) > self.capacity:
            self.memories.pop(0)

    def get_all_memories(self) -> List[str]:
        return self.memories

    def clear_memories(self) -> None:
        self.memories.clear()

    def search_memories(self, keyword: str) -> List[str]:
        return [memory for memory in self.memories if keyword.lower() in memory.lower()]