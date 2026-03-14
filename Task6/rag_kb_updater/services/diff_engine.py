from domain.models import Diff, FileInfo

class DiffEngine:
    def diff(self, old: dict[str, FileInfo], new: dict[str, FileInfo]) -> Diff:
        old_paths = set(old.keys())
        new_paths = set(new.keys())

        added = sorted(new_paths - old_paths)
        deleted = sorted(old_paths - new_paths)

        modified: list[str] = []
        for p in (old_paths & new_paths):
            if old[p].sha256 != new[p].sha256:
                modified.append(p)

        return Diff(added=added, modified=sorted(modified), deleted=deleted)