"""下载 Google DeepMind MuJoCo Menagerie 的 franka_emika_panda 目录（仅该目录）。

本机无 git，且 raw.githubusercontent.com 在 Python 下 DNS 解析不稳定，
故：
  1. 用 GitHub REST API 递归获取文件清单（路径 / 字节数 / git blob sha）；
  2. 依次尝试多个可达源下载（直连 raw -> ghfast.top 镜像 -> gh-proxy 镜像 -> API raw）；
  3. 对每个文件做 **字节数 + git blob SHA1** 双重校验，任何不一致即报错。

不修改任何官方文件内容。
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

REPO = "google-deepmind/mujoco_menagerie"
SUBDIR = "franka_emika_panda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "assets", "mujoco_menagerie", SUBDIR)
API = f"https://api.github.com/repos/{REPO}/contents/{SUBDIR}"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/{SUBDIR}"
SOURCES = [
    lambda rel: f"{RAW}/{rel}",
    lambda rel: f"https://ghfast.top/{RAW}/{rel}",
    lambda rel: f"https://gh-proxy.com/{RAW}/{rel}",
    lambda rel: f"{API}/{rel}",  # 需要 Accept: raw
]


def get_json(url, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mujoco-setup"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"listing failed: {url}: {last}")


def git_blob_sha1(blob: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(blob))
    h.update(blob)
    return h.hexdigest()


def fetch(rel, want_size, want_sha, retries=3):
    last = None
    for attempt in range(retries):
        for idx, src in enumerate(SOURCES):
            url = src(rel)
            headers = {"User-Agent": "mujoco-setup"}
            if url.startswith("https://api.github.com"):
                headers["Accept"] = "application/vnd.github.raw"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r:
                    blob = r.read()
                if len(blob) != want_size:
                    raise ValueError(f"size {len(blob)} != {want_size}")
                got = git_blob_sha1(blob)
                if want_sha and got != want_sha:
                    raise ValueError(f"sha {got} != {want_sha}")
                return blob, idx
            except Exception as exc:
                last = f"{url} -> {exc}"
                continue
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed for {rel}: {last}")


def walk(rel=""):
    items = get_json(API + (f"/{rel}" if rel else ""))
    count = 0
    for it in items:
        name = it["name"]
        rel_path = f"{rel}/{name}" if rel else name
        if it["type"] == "dir":
            count += walk(rel_path)
        elif it["type"] == "file":
            out = os.path.join(DEST, rel_path.replace("/", os.sep))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if (os.path.exists(out) and os.path.getsize(out) == it["size"]
                    and git_blob_sha1(open(out, "rb").read()) == it["sha"]):
                print(f"  [skip] {rel_path}", flush=True)
                count += 1
                continue
            blob, src_idx = fetch(rel_path, it["size"], it["sha"])
            with open(out, "wb") as f:
                f.write(blob)
            print(f"  {rel_path}  {len(blob)} bytes  sha_ok  src={src_idx}", flush=True)
            count += 1
    return count


def main():
    print("目标目录:", DEST)
    n = walk()
    print(f"文件数: {n}")
    total = 0
    for dirpath, _, files in os.walk(DEST):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            total += os.path.getsize(p)
            print(f"{os.path.getsize(p):>9}  {os.path.relpath(p, DEST)}")
    print(f"总字节: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
