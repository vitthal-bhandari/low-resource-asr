"""
Script to download Mozilla Common Voice Spontaneous Speech datasets.

Downloads data from Mozilla Data Collective API and organizes it into the expected format:
    data/mozilla_speech_data/
        {lang_code}/
            train/
                audios/
                metadata.csv
            validation/
                audios/
                metadata.csv

Usage:
    # Download train/dev data for all languages
    uv run python -m src.data.download --split train-dev
    
    # Download test data
    uv run python -m src.data.download --split test
    
    # Download both
    uv run python -m src.data.download --split all
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import config

# All 21 languages in the Mozilla Spontaneous Speech dataset
LANGUAGES = {
    # Africa
    "bxk": "Bukusu",
    "cgg": "Chiga",
    "kcn": "Nubi",
    "koo": "Konzo",
    "led": "Lendu",
    "lke": "Kenyi",
    "lth": "Thur",
    "ruc": "Ruuli",
    "rwm": "Amba",
    "ttj": "Rutoro",
    "ukv": "Kuku",
    # Americas
    "hch": "Wixárika",
    "meh": "Southwestern Tlaxiaco Mixtec",
    "mmc": "Michoacán Mazahua",
    "top": "Papantla Totonac",
    "tob": "Toba Qom",
    # Europe
    "aln": "Gheg Albanian",
    "el-CY": "Cypriot Greek",
    "sco": "Scots",
    # Asia
    "bew": "Betawi",
    "pne": "Western Penan",
}

# 5 unseen languages (test only, no training data)
UNSEEN_LANGUAGES = {
    "ady": "Adyghe",
    "kbd": "Kabardian",
    "bas": "Basaa",
    "qxp": "Puno Quechua",
    "ush": "Ushojo",
}


def get_download_url(dataset_id: str, api_key: str) -> str:
    """
    Get the download URL from Mozilla Data Collective API.
    
    Args:
        dataset_id: The dataset ID from MDC
        api_key: Your MDC API key
        
    Returns:
        The download URL for the dataset
        
    Raises:
        requests.HTTPError: If the API request fails
        ValueError: If the response doesn't contain a download URL
    """
    url = config.get_mdc_download_url(dataset_id)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    download_url = data.get("downloadUrl")
    
    if not download_url:
        raise ValueError(f"No download URL in response: {data}")
    
    return download_url


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> Path:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        desc: Description for progress bar
        
    Returns:
        Path to the downloaded file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    return output_path


def extract_tarball(tarball_path: Path, extract_dir: Path) -> Path:
    """
    Extract a tar.gz file.
    
    Args:
        tarball_path: Path to the tarball
        extract_dir: Directory to extract to
        
    Returns:
        Path to the extraction directory
    """
    print(f"Extracting {tarball_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tarball_path, "r:gz") as tar:
        # Get total members for progress
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, extract_dir)
    
    return extract_dir


def organize_extracted_data(
    extracted_dir: Path,
    output_dir: Path,
    split_type: str,
) -> dict[str, bool]:
    """
    Organize extracted data into the expected folder structure.
    
    The Mozilla tarball structure is expected to be:
        {lang_code}/
            train/
                audios/
                metadata.csv
            dev/ or validation/
                audios/
                metadata.csv
    
    Args:
        extracted_dir: Directory where tarball was extracted
        output_dir: Target output directory (data/mozilla_speech_data/)
        split_type: "train-dev" or "test"
        
    Returns:
        Dictionary mapping language codes to success status
    """
    results = {}
    
    # Find all language directories in extracted data
    # The structure might have a top-level directory from the tarball
    search_dirs = [extracted_dir]
    
    # Check if there's a single top-level directory
    subdirs = [d for d in extracted_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not any(subdirs[0].name == lang for lang in LANGUAGES):
        search_dirs = [subdirs[0]]
    
    for search_dir in search_dirs:
        for item in search_dir.iterdir():
            if not item.is_dir():
                continue
            
            lang_code = item.name
            
            # Check if this is a known language
            all_langs = {**LANGUAGES, **UNSEEN_LANGUAGES}
            if lang_code not in all_langs:
                print(f"  Skipping unknown directory: {lang_code}")
                continue
            
            print(f"  Processing {lang_code} ({all_langs[lang_code]})...")
            
            target_lang_dir = output_dir / lang_code
            target_lang_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Copy/move the data
                for split in ["train", "dev", "validation", "test"]:
                    source_split = item / split
                    if source_split.exists():
                        # Normalize "dev" to "validation"
                        target_split_name = "validation" if split == "dev" else split
                        target_split = target_lang_dir / target_split_name
                        
                        if target_split.exists():
                            shutil.rmtree(target_split)
                        
                        shutil.copytree(source_split, target_split)
                        print(f"    {split} -> {target_split_name}: OK")
                
                results[lang_code] = True
                
            except Exception as e:
                print(f"    ERROR: {e}")
                results[lang_code] = False
    
    return results


def download_dataset(
    split: str,
    output_dir: Path,
    api_key: str,
    keep_tarball: bool = False,
) -> dict[str, bool]:
    """
    Download and extract a dataset split.
    
    Args:
        split: "train-dev" or "test"
        output_dir: Output directory for organized data
        api_key: Mozilla Data Collective API key
        keep_tarball: Whether to keep the downloaded tarball
        
    Returns:
        Dictionary mapping language codes to success status
    """
    # Determine dataset ID
    if split == "train-dev":
        dataset_id = config.mdc_train_dev_dataset_id
        tarball_name = "mdc_train_dev.tar.gz"
    elif split == "test":
        dataset_id = config.mdc_test_dataset_id
        tarball_name = "mdc_test.tar.gz"
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train-dev' or 'test'")
    
    print(f"\n{'='*60}")
    print(f"Downloading {split} data")
    print(f"{'='*60}")
    
    # Get download URL
    print("Getting download URL from MDC API...")
    try:
        download_url = get_download_url(dataset_id, api_key)
        print("  Download URL obtained")
    except requests.HTTPError as e:
        print(f"  ERROR: API request failed - {e}")
        print("  Check that your MDC_API_KEY is valid")
        return {}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}
    
    # Download tarball
    tarball_path = output_dir / tarball_name
    print(f"\nDownloading to {tarball_path}...")
    try:
        download_file(download_url, tarball_path, desc=f"Downloading {split}")
    except Exception as e:
        print(f"  ERROR: Download failed - {e}")
        return {}
    
    # Extract
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_dir = Path(temp_dir) / "extracted"
        try:
            extract_tarball(tarball_path, extract_dir)
        except Exception as e:
            print(f"  ERROR: Extraction failed - {e}")
            return {}
        
        # Organize into expected structure
        print("\nOrganizing data...")
        results = organize_extracted_data(extract_dir, output_dir, split)
    
    # Cleanup tarball
    if not keep_tarball and tarball_path.exists():
        print(f"\nRemoving tarball...")
        tarball_path.unlink()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download Mozilla Common Voice Spontaneous Speech datasets"
    )
    parser.add_argument(
        "--split",
        choices=["train-dev", "test", "all"],
        default="train-dev",
        help="Which split to download (default: train-dev)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/mozilla_speech_data)",
    )
    parser.add_argument(
        "--keep-tarball",
        action="store_true",
        help="Keep the downloaded tarball after extraction",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all available languages and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_languages:
        print("Training languages (21):")
        for code, name in sorted(LANGUAGES.items()):
            print(f"  {code}: {name}")
        print("\nUnseen/test-only languages (5):")
        for code, name in sorted(UNSEEN_LANGUAGES.items()):
            print(f"  {code}: {name}")
        return
    
    # Validate config
    missing = config.validate()
    if missing:
        print("ERROR: Missing required configuration:")
        for key in missing:
            print(f"  - {key}")
        print("\nPlease create a .env file with your API key:")
        print("  cp .env.example .env")
        print("  # Then edit .env and add your MDC_API_KEY")
        return
    
    # Set output directory
    output_dir = args.output_dir or config.mozilla_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Mozilla Common Voice Spontaneous Speech Dataset Downloader")
    print(f"Output directory: {output_dir}")
    
    all_results = {}
    
    # Download requested splits
    if args.split in ["train-dev", "all"]:
        results = download_dataset(
            "train-dev",
            output_dir,
            config.mdc_api_key,
            args.keep_tarball,
        )
        all_results.update(results)
    
    if args.split in ["test", "all"]:
        results = download_dataset(
            "test",
            output_dir,
            config.mdc_api_key,
            args.keep_tarball,
        )
        all_results.update(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    if all_results:
        successful = sum(all_results.values())
        print(f"Languages processed: {successful}/{len(all_results)}")
        
        failed = [lang for lang, success in all_results.items() if not success]
        if failed:
            print(f"Failed: {', '.join(failed)}")
    else:
        print("No data was downloaded. Check errors above.")
    
    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()
