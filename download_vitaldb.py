# Standalone script to download VitalDB .mat files from Google Drive.
# Run in a CMD window - survives VS Code / Claude Code closure.
# Re-run to resume if interrupted.

import gdown
import json
import os
import time
import sys

OUTPUT_DIR = r"F:\Projet\Script\PPGAge\data\pulsedb_segments\PulseDB_Vital"
DOWNLOAD_LIST = r"F:\Projet\Script\PPGAge\data\download_list.json"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DOWNLOAD_LIST) as f:
        to_download = json.load(f)  # list of [filename, gdrive_id]

    # Skip already downloaded
    existing = set(f.replace('.mat', '') for f in os.listdir(OUTPUT_DIR) if f.endswith('.mat'))
    remaining = [(name, fid) for name, fid in to_download if name not in existing]

    print(f"Total to download: {len(to_download)}")
    print(f"Already have: {len(to_download) - len(remaining)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    failed = []
    for i, (name, fid) in enumerate(remaining):
        out_path = os.path.join(OUTPUT_DIR, f"{name}.mat")
        print(f"[{i+1}/{len(remaining)}] {name}...", end=" ", flush=True)
        try:
            gdown.download(id=fid, output=out_path, quiet=True)
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"{size_mb:.0f} MB")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)
            if "quota" in str(e).lower() or "too many" in str(e).lower():
                print("\nRate limited by Google Drive. Waiting 60s before retry...")
                time.sleep(60)
                # Retry once
                try:
                    gdown.download(id=fid, output=out_path, quiet=True)
                    size_mb = os.path.getsize(out_path) / 1024 / 1024
                    print(f"  Retry OK: {size_mb:.0f} MB")
                    failed.pop()
                except Exception as e2:
                    print(f"  Retry also failed: {e2}")
                    print("  Stopping. Re-run this script later to resume.")
                    break

    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mat')])
    print(f"\nDone. Total files: {total}")
    if failed:
        print(f"Failed: {failed}")
    print("Re-run this script to resume any remaining downloads.")


if __name__ == "__main__":
    main()
