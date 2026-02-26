#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pwd
import argparse
import datetime
from collections import defaultdict

def human_bytes(n: int) -> str:
    """Convert bytes to human-readable string."""
    units = ['B', 'K', 'M', 'G', 'T', 'P']
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f}{u}"
        size /= 1024

def safe_stat(path: str):
    try:
        return os.lstat(path)
    except Exception:
        return None

def get_birth_ctime_mtime(st) -> float:
    bt = getattr(st, "st_birthtime", None)
    if isinstance(bt, (int, float)):
        return float(bt)
    if hasattr(st, "st_ctime"):
        return float(st.st_ctime)
    return float(st.st_mtime)

def du(path: str) -> int:
    """Disk usage in bytes (recursive for directories)."""
    st = safe_stat(path)
    if st is None:
        return 0
    if not os.path.isdir(path) or os.path.islink(path):
        return int(st.st_size)
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for name in files:
            p = os.path.join(root, name)
            stf = safe_stat(p)
            if stf is not None:
                total += int(stf.st_size)
    return total

def owner_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid:{uid}"

def scan_tmp(tmp_dir: str, filter_user: str = None):
    items = []
    try:
        with os.scandir(tmp_dir) as it:
            for entry in it:
                st = safe_stat(entry.path)
                if st is None:
                    continue
                user = owner_name(st.st_uid)
                if filter_user and user != filter_user:
                    continue
                ts = get_birth_ctime_mtime(st)
                size = du(entry.path) if entry.is_dir(follow_symlinks=False) else int(st.st_size)
                items.append({
                    "user": user,
                    "path": entry.path,
                    "name": entry.name,
                    "is_dir": entry.is_dir(follow_symlinks=False),
                    "size": size,
                    "time": ts,
                })
    except PermissionError:
        print(f"Permission denied when scanning {tmp_dir}", file=sys.stderr)
        sys.exit(1)

    return items

def print_summary(items):
    user_sizes = defaultdict(int)
    for it in items:
        user_sizes[it["user"]] += it["size"]

    total_size = sum(user_sizes.values())
    print(f"# User summary for /tmp | total={human_bytes(total_size)}")
    print("-" * 60)
    print(f"{'User':10} {'Total Size':>12}")
    for user, size in sorted(user_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"{user:10} {human_bytes(size):>12}")

def main():
    parser = argparse.ArgumentParser(description="Analyze /tmp usage by users.")
    parser.add_argument("--tmp-dir", default="/tmp", help="Directory to scan (default: /tmp)")
    parser.add_argument("--user", help="Only show entries belonging to this user (default: all)")
    parser.add_argument("--summary", action="store_true", help="Show summary by user instead of detailed list")
    args = parser.parse_args()

    if not os.path.isdir(args.tmp_dir):
        print(f"{args.tmp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    items = scan_tmp(args.tmp_dir, filter_user=args.user)
    if args.summary:
        print_summary(items)
    else:
        for it in items:
            print(it)

if __name__ == "__main__":
    main()
