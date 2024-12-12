from utils import repo_manager
from utils.gitw import Repo

linux = Repo(repo_manager.get("git@github.com:torvalds/linux"))
linuxstable = Repo(repo_manager.get("git@github.com:gregkh/linux"))
asahi = Repo(repo_manager.get("git@github.com:asahilinux/linux"))
lkl = Repo(repo_manager.get("git@github.com:lkl/linux"))
oh = Repo(repo_manager.get("git@gitee.com:openharmony/kernel_linux_5.10"))
redis = Repo(repo_manager.get("git@github.com:redis/redis"))
hiredis = Repo(repo_manager.get("git@github.com:redis/hiredis"))
birdisle = Repo(repo_manager.get("git@github.com:bmerry/birdisle"))
target2repo = {
    "linux-v6.3-rc5": linux,
    "linux-v6.2.9": linuxstable,
    "linux-v5.15.105": linuxstable,
    "linux-asahi-asahi-6.3-9": asahi,
    "linux-lkl-970883c3": lkl,
    "linux-oh-2f4ecbf3": oh,
    "redis-7.0.10": redis,
    "redis-5.0.14": redis,
    "redis-birdisle-06e93987": birdisle,
}
vul_sig_repos = {
    "redis": redis,
    "linux": linux,
    "hiredis": hiredis,
}
