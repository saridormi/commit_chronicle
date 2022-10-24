from dataclasses import dataclass
from typing import Optional, Set, Tuple


@dataclass
class CloneGroup:
    clone_root: Optional[Tuple[int, int]]
    clones: Set[Tuple[int, int]]

    def get_ids_to_drop(self, include_root: bool = True):
        all_clones = [clone[1] for clone in self.clones]
        if include_root and self.clone_root:
            all_clones.append(self.clone_root[1])
        return all_clones
