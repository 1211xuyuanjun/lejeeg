"""
Data download utilities for EEG pre-training dataset.

Supports two download methods:
1. Hugging Face Hub (hf_hub_download)
2. wget (shell command)
"""

import os
from os.path import join as pjoin
from typing import Optional
from data_number import SUBSET_TRAIN, SUBSET_VAL


def download_file(
    remote_file: str,
    local_path: str,
    repo_id: str = "brain-bzh/reve-dataset",
    repo_type: str = "dataset",
    method: str = "huggingface",
    mirror: Optional[str] = "https://hf-mirror.com",
    force_download: bool = False,
) -> None:
    """
    Download a single file from Hugging Face Hub or using wget.
    
    Args:
        remote_file: Remote file path relative to repo root, e.g., "data/recording_-_eeg_-_100.npy"
        local_path: Local file path to save
        repo_id: Hugging Face repo ID
        repo_type: "dataset" or "model"
        method: "huggingface" or "wget"
        mirror: Mirror URL for Hugging Face Hub (e.g., "https://hf-mirror.com")
        force_download: Force download even if file exists
    """
    # Create directory if needed
    local_dir = os.path.dirname(local_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    
    # Skip if file already exists
    if os.path.exists(local_path) and not force_download:
        print(f"File already exists: {local_path}")
        return
    
    print(f"Downloading {remote_file} to {local_path}")
    if method == "huggingface":
        try:
            from huggingface_hub import hf_hub_download
            
            # Set mirror if provided
            if mirror:
                print(f"method: {method}")
                print(f"mirror: {mirror}")

                os.environ["HF_ENDPOINT"] = mirror
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_file,
                repo_type=repo_type,
                force_download=force_download,
                local_dir=os.path.dirname(local_path) if os.path.dirname(local_path) else None,
            )
            print(f"Successfully downloaded to: {downloaded_path}")
            
        except ImportError:
            print("huggingface_hub not found, falling back to wget")
            download_file(remote_file, local_path, repo_id, repo_type, method="wget", mirror=mirror, force_download=force_download)
    
    elif method == "wget":
        import subprocess
        import sys
        
        # Build wget URL
        if mirror:
            url = f"{mirror}/datasets/{repo_id}/resolve/main/{remote_file}?download=true"
        else:
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_file}?download=true"
        
        # Run wget
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "wget"],
                check=False,
                capture_output=True,
            )
            import wget
            wget.download(url, out=local_path)
            print(f"\nSuccessfully downloaded to: {local_path}")
        except (ImportError, subprocess.CalledProcessError):
            # Try using shell wget
            try:
                subprocess.run(
                    ["wget", "-O", local_path, url],
                    check=True,
                )
                print(f"Successfully downloaded to: {local_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(
                    "Failed to download. Please install huggingface_hub: "
                    "pip install huggingface_hub"
                )
    
    else:
        raise ValueError(f"Unknown download method: {method}")


def download_eeg_recording(
    recording_index: int,
    data_dir: str,
    method: str = "huggingface",
    mirror: Optional[str] = "https://hf-mirror.com",
    force_download: bool = False,
    download_eeg: bool = True,
    download_positions: bool = False,
    download_stats: bool = False,
) -> None:
    """
    Download EEG recording files.
    
    Args:
        recording_index: Recording index, e.g., 100 for recording_-_eeg_-_100.npy
        data_dir: Local data directory (should contain "recordings" subdir)
        method: "huggingface" or "wget"
        mirror: Mirror URL for Hugging Face Hub
        force_download: Force download even if files exist
        download_eeg: Whether to download EEG data
        download_positions: Whether to download position data
        download_stats: Whether to download stats data
    """
    recordings_dir = pjoin(data_dir, "recordings")
    
    files_to_download = []
    if download_eeg:
        files_to_download.append((
            f"data/recording_-_eeg_-_{recording_index}.npy",
            pjoin(recordings_dir, f"recording_-_eeg_-_{recording_index}.npy"),
        ))
    if download_positions:
        files_to_download.append((
            f"positions/recording_-_positions_-_{recording_index}.npy",
            pjoin(recordings_dir, f"recording_-_positions_-_{recording_index}.npy"),
        ))
    if download_stats:
        files_to_download.append((
            f"stats/recording_-_stats_-_{recording_index}.npy",
            pjoin(recordings_dir, f"recording_-_stats_-_{recording_index}.npy"),
        ))
    
    for remote_file, local_path in files_to_download:
        download_file(
            remote_file=remote_file,
            local_path=local_path,
            method=method,
            mirror=mirror,
            force_download=force_download,
        )


def download_csv_metadata(
    data_dir: str,
    method: str = "huggingface",
    mirror: Optional[str] = "https://hf-mirror.com",
    force_download: bool = False,
) -> None:
    """
    Download CSV metadata files.
    
    Args:
        data_dir: Local data directory (should contain "csv_recordings" subdir)
        method: "huggingface" or "wget"
        mirror: Mirror URL for Hugging Face Hub
        force_download: Force download even if files exist
    """
    csv_dir = pjoin(data_dir, "csv_recordings")
    
    csv_files = [
        ("df_big.csv", pjoin(csv_dir, "df_big.csv")),
        ("df_corrected.csv", pjoin(csv_dir, "df_corrected.csv")),
        ("df_stats.csv", pjoin(csv_dir, "df_stats.csv")),
    ]
    
    for remote_file, local_path in csv_files:
        download_file(
            remote_file=remote_file,
            local_path=local_path,
            method=method,
            mirror=mirror,
            force_download=force_download,
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download EEG dataset files")
    parser.add_argument("--data-dir", type=str, default="./data", help="Local data directory")
    parser.add_argument("--recording-index", type=int, nargs='+', help="Download specific recording index (can be multiple, e.g., 100 101 102)")
    parser.add_argument("--download-csv", action="store_true", help="Download CSV metadata")
    parser.add_argument("--method", type=str, default="huggingface", choices=["huggingface", "wget"], help="Download method")
    parser.add_argument("--no-mirror", action="store_true", help="Disable Hugging Face mirror")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--download-positions", action="store_true", help="Download position files (default: False)")
    parser.add_argument("--download-stats", action="store_true", help="Download stats files (default: False)")
    
    args = parser.parse_args()
    
    mirror = None if args.no_mirror else "https://hf-mirror.com"
    
    if args.download_csv:
        print("Downloading CSV metadata...")
        download_csv_metadata(
            data_dir=args.data_dir,
            method=args.method,
            mirror=mirror,
            force_download=args.force,
        )
    
    if args.recording_index is not None:
        args.recording_index = SUBSET_TRAIN
        # 支持多个索引
        for idx in args.recording_index:
            print(f"Downloading recording {idx}...")
            download_eeg_recording(
                recording_index=idx,
                data_dir=args.data_dir,
                method=args.method,
                mirror=mirror,
                force_download=args.force,
                download_eeg=True,
                download_positions=args.download_positions,
                download_stats=args.download_stats,
            )
    
    if not args.download_csv and args.recording_index is None:
        print("Please specify either --download-csv or --recording-index")