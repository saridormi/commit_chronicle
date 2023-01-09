from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple


@dataclass
class CloneGroup:
    clone_root: Optional[Tuple[int, int]]
    clones: Set[Tuple[int, int]]

    def get_ids_to_drop(self, ids_map: Optional[Dict[int, int]], include_root: bool = True):
        all_clones = [clone[1] for clone in self.clones]
        if include_root and self.clone_root:
            all_clones.append(self.clone_root[1])
        if ids_map:
            all_clones = [ids_map[clone] for clone in all_clones]
        return all_clones
