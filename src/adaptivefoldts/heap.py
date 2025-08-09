import heapq
from typing import List, Tuple, Any

class AdaptiveHeap:
    def __init__(self):
        self._heap: List[Tuple[float, Any]] = []

    def push(self, priority: float, item: Any):
        """Adiciona item com a prioridade dada."""
        heapq.heappush(self._heap, (priority, item))

    def pop(self) -> Tuple[float, Any]:
        """Remove e retorna o item com a menor prioridade."""
        return heapq.heappop(self._heap)

    def peek(self) -> Tuple[float, Any]:
        """Retorna o item com menor prioridade sem remover."""
        return self._heap[0]

    def __len__(self):
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0
