import fcntl
import orjson
import multiprocessing as mp
import os
import shutil
import tempfile
from typing import Optional
from loguru import logger
from git import Repo

POOL_FILE = os.path.join(os.path.dirname(__file__), "repo_pool.json")
REPOSITORY_BASE = "/data/repos"


class NoRepoError(Exception):
    def __init__(self, repo):
        self.repo = repo

    def __str__(self):
        return "No such repo: {}".format(self.repo)


class LRURepoPoolSafe:
    def __init__(self, base, max_size=4000):
        self.max_size = max_size
        self.lrucache: dict[str, int] = {}
        self.badcache: set[str] = set()
        self.lock = mp.Lock()
        self.base = base

        try:
            self._load()
            self._validate()
        except FileNotFoundError:
            pass

    def _validate(self):
        for key in list(self.lrucache.keys()):
            path = os.path.join(self.base, key)
            if not os.path.exists(path):
                del self.lrucache[key]

    def has(self, git_url):
        key = self._digest(git_url)
        return key in self.lrucache or key in self.badcache

    def evict(self, git_url):
        key = self._digest(git_url)
        path = os.path.join(self.base, key)
        lock = os.path.join(self.base, key + ".lock")
        if key in self.lrucache:
            logger.info("Evict {}", git_url)
            del self.lrucache[key]
            if os.path.exists(path):
                shutil.rmtree(path)
            if os.path.exists(lock):
                os.unlink(lock)
            self._dump()

    def forget(self, git_url):
        key = self._digest(git_url)
        with self.lock:
            if key in self.badcache:
                self.badcache.remove(key)
                self._dump()

    def get(self, git_url) -> Optional[Repo]:
        key = self._digest(git_url)
        path = os.path.join(self.base, key)
        lock = os.path.join(self.base, key + ".lock")

        with self.lock:
            # The repo is already cloned and in the cache
            if key in self.lrucache:
                self.lrucache[key] += 1
                self._dump()
                return Repo(path)

            # The url is bad
            if key in self.badcache:
                return None

            if len(self.lrucache) >= self.max_size:
                # remove the least used item
                least_used = min(self.lrucache.items(), key=lambda x: x[1])
                least_key = least_used[0]
                least_path = os.path.join(self.base, key)
                least_lock = os.path.join(self.base, key + ".lock")
                shutil.rmtree(least_path)
                if os.path.exists(least_key):
                    os.unlink(least_lock)
                del self.lrucache[least_key]
                self._dump()

        # acquire a file lock
        # if the lock is acquired, we are the only process that is cloning the repo
        # otherwise, we wait for the other process to finish cloning
        with open(lock, "wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # The repo has been cloned by another process or failed
                # We can just open it
                if os.path.exists(path):
                    inst = Repo(path)
                    with self.lock:
                        if key not in self.lrucache:
                            self.lrucache[key] = 1
                        else:
                            self.lrucache[key] += 1
                        self._dump()
                    return inst

                with self.lock:
                    # The url is bad
                    if key in self.badcache:
                        return None

                # Clone the repo
                try:
                    logger.info("Cloning from {}", git_url)
                    inst = Repo.clone_from(git_url, path)
                except Exception as e:
                    # Clone failed, bad git url
                    logger.error(e)
                    with self.lock:
                        self.badcache.add(key)
                    return None

                # Clone succeeded
                with self.lock:
                    if key not in self.lrucache:
                        self.lrucache[key] = 1
                    else:
                        self.lrucache[key] += 1
                    self._dump()
                    return Repo(path)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _digest(self, git_url: str):
        git_url = git_url.strip().lower()
        git_url = git_url.rstrip("/")
        if git_url.endswith(".git"):
            git_url = git_url[:-4]
        if git_url.startswith("https://"):
            group, repo = git_url.split("/")[-2:]
        else:
            group, repo = git_url.split(":")[-1].split("/")
        return f"{group}+{repo}"

    def _dump(self, file=POOL_FILE):
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(
                orjson.dumps(
                    {
                        "lrucache": self.lrucache,
                        "badcache": list(self.badcache),
                    }
                )
            )
            os.rename(f.name, file)

    def _load(self, file=POOL_FILE):
        if os.path.exists(file):
            obj = orjson.loads(open(file, "rb").read())
            self.lrucache = obj["lrucache"]
            self.badcache = set(obj["badcache"])


pool = LRURepoPoolSafe(REPOSITORY_BASE, 5000)


def get(git_url) -> Repo:
    repo = pool.get(git_url)
    if repo is None:
        raise NoRepoError(git_url)
    return repo
