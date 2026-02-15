"""
Script to download and organize Mozilla Common Voice Spontaneous Speech datasets.

Downloads train/dev data only from Mozilla Data Collective API and organizes into:
    data/mozilla_speech_data/
        shared_train_validation_audios/   # All train/validation audio files
        {lang_code}/                      # TSV files for each language
            ss-corpus-{lang}.tsv          # Train/validation metadata
            ss-reported-audios-{lang}.tsv # Reported audios (if present)

Test data is not downloaded. Validation can later be split into test/val via
scripts/split_dev_to_test_val.py.

Usage:
    uv run python -m src.data.download

Prerequisites:
    Set MDC_API_KEY in .env file
"""

import shutil
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
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


def get_download_url(dataset_id: str, api_key: str) -> str:
    """
    Get the download URL from Mozilla Data Collective API.
    
    Args:
        dataset_id: The dataset ID from MDC
        api_key: Your MDC API key
        
    Returns:
        The download URL for the dataset
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
    """Download a file with progress bar."""
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
    """Extract a tar/tar.gz file with progress."""
    print(f"Extracting {tarball_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine compression
    if tarball_path.suffix == ".gz" or tarball_path.name.endswith(".tar.gz"):
        mode = "r:gz"
    else:
        mode = "r"
    
    with tarfile.open(tarball_path, mode) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="  Extracting"):
            tar.extract(member, extract_dir)
    
    return extract_dir


def extract_lang_code_from_dirname(dirname: str) -> str | None:
    """Extract language code from directory names like 'sps-corpus-1.0-2025-09-05-aln'."""
    if dirname in LANGUAGES:
        return dirname
    
    if dirname.startswith("sps-corpus-"):
        if dirname.endswith("-el-CY"):
            return "el-CY"
        last_segment = dirname.split("-")[-1]
        if last_segment in LANGUAGES:
            return last_segment
    
    return None


def organize_train_validation_data(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Organize train/validation data.
    
    Args:
        source_dir: Extracted train/validation data directory
        output_dir: Target output directory (mozilla_speech_data/)
        
    Returns:
        Dictionary with counts: {lang_code: audio_count}
    """
    audio_counts = {}
    shared_audio_dir = output_dir / "shared_train_validation_audios"
    shared_audio_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nOrganizing train/validation data...")
    
    # Find the actual data directory (may be nested)
    search_dir = source_dir
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and extract_lang_code_from_dirname(subdirs[0].name) is None:
        # There's a wrapper directory
        search_dir = subdirs[0]
    
    for item in sorted(search_dir.iterdir()):
        if not item.is_dir():
            continue
        
        lang_code = extract_lang_code_from_dirname(item.name)
        if lang_code is None:
            continue
        
        print(f"  [{lang_code}] {LANGUAGES[lang_code]}")
        
        lang_dir = output_dir / lang_code
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy TSV files
        for tsv_file in item.glob("*.tsv"):
            target_tsv = lang_dir / tsv_file.name
            shutil.copy(tsv_file, target_tsv)
            print(f"    Copied {tsv_file.name}")
        
        # Copy audio files to shared directory
        audio_dir = item / "audios"
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.mp3"))
            audio_counts[lang_code] = len(audio_files)
            
            for audio_file in tqdm(audio_files, desc=f"    Copying audios", leave=False):
                target_audio = shared_audio_dir / audio_file.name
                if not target_audio.exists():
                    shutil.copy(audio_file, target_audio)
            
            print(f"    Copied {len(audio_files)} audio files")
        else:
            audio_counts[lang_code] = 0
            print(f"    WARNING: No audios directory found")
    
    return audio_counts


def organize_test_data(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """
    Organize test data (21 languages only, skip unseen).
    
    Args:
        source_dir: Extracted test data directory
        output_dir: Target output directory (mozilla_speech_data/)
        
    Returns:
        Dictionary with counts: {lang_code: audio_count}
    """
    audio_counts = {}
    shared_audio_dir = output_dir / "shared_test_audios"
    shared_audio_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nOrganizing test data...")
    
    # Find the multilingual-general directory (may be nested)
    multilingual_dir = source_dir / "multilingual-general"
    actual_source = source_dir
    
    if not multilingual_dir.exists():
        for subdir in source_dir.iterdir():
            if subdir.is_dir():
                potential = subdir / "multilingual-general"
                if potential.exists():
                    multilingual_dir = potential
                    actual_source = subdir
                    break
    
    if not multilingual_dir.exists():
        print("  ERROR: multilingual-general directory not found")
        return audio_counts
    
    # Copy test TSV files for each language
    for tsv_file in sorted(multilingual_dir.glob("*.tsv")):
        lang_code = tsv_file.stem
        
        if lang_code not in LANGUAGES:
            print(f"  Skipping unseen language: {lang_code}")
            continue
        
        lang_dir = output_dir / lang_code
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        target_tsv = lang_dir / f"test-{lang_code}.tsv"
        shutil.copy(tsv_file, target_tsv)
        print(f"  [{lang_code}] Copied test TSV")
    
    # Copy test audio files to shared directory (21 languages only)
    audio_dir = actual_source / "audios"
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.mp3"))
        copied = 0
        
        for audio_file in tqdm(audio_files, desc="  Copying test audios"):
            parts = audio_file.stem.split("-")
            if len(parts) >= 3:
                if "el-CY" in audio_file.name:
                    file_lang = "el-CY"
                else:
                    file_lang = parts[2]
                
                if file_lang in LANGUAGES:
                    target_audio = shared_audio_dir / audio_file.name
                    if not target_audio.exists():
                        shutil.copy(audio_file, target_audio)
                    copied += 1
                    audio_counts[file_lang] = audio_counts.get(file_lang, 0) + 1
        
        print(f"  Copied {copied} audio files (21 languages only)")
    else:
        print("  WARNING: No audios directory found")
    
    return audio_counts


def verify_counts(output_dir: Path) -> bool:
    """Verify that TSV file counts match audio file counts."""
    print("\n" + "=" * 60)
    print("Verifying data integrity (train/val only)...")
    print("=" * 60)
    
    all_match = True
    
    train_val_audio_dir = output_dir / "shared_train_validation_audios"
    train_val_audio_files = set(f.name for f in train_val_audio_dir.glob("*.mp3")) if train_val_audio_dir.exists() else set()
    
    print(f"\nShared audio directory:")
    print(f"  Train/Validation: {len(train_val_audio_files)} files")
    
    print(f"\nPer-language verification:")
    print(f"{'Lang':<8} {'Train':<8} {'Val':<8} {'Status'}")
    print("-" * 40)
    
    total_train = 0
    total_val = 0
    tsv_train_val_files = set()
    
    for lang_code in sorted(LANGUAGES.keys()):
        lang_dir = output_dir / lang_code
        if not lang_dir.exists():
            print(f"{lang_code:<8} {'N/A':<8} {'N/A':<8} MISSING")
            all_match = False
            continue
        
        corpus_tsv = lang_dir / f"ss-corpus-{lang_code}.tsv"
        train_count = 0
        val_count = 0
        
        if corpus_tsv.exists():
            df = pd.read_csv(corpus_tsv, sep="\t")
            if "split" in df.columns and "audio_file" in df.columns:
                train_count = len(df[df["split"] == "train"])
                val_count = len(df[df["split"] == "dev"])
                for _, row in df.iterrows():
                    if pd.notna(row.get("audio_file")):
                        tsv_train_val_files.add(row["audio_file"])
        
        total_train += train_count
        total_val += val_count
        print(f"{lang_code:<8} {train_count:<8} {val_count:<8} OK")
    
    print("-" * 40)
    print(f"{'TOTAL':<8} {total_train:<8} {total_val:<8}")
    
    print(f"\nFile matching verification:")
    tsv_total = len(tsv_train_val_files)
    audio_total = len(train_val_audio_files)
    missing_audios = tsv_train_val_files - train_val_audio_files
    
    print(f"  Train/Val TSV entries: {tsv_total}")
    print(f"  Train/Val audio files: {audio_total}")
    if missing_audios:
        print(f"  WARNING: {len(missing_audios)} audio files listed in TSV but not found")
        all_match = False
    else:
        print(f"  MATCH: All TSV audio references found")
    
    return all_match


def main():
    """Main entry point - download and organize all data."""
    print("=" * 60)
    print("Mozilla Common Voice Spontaneous Speech Dataset Downloader")
    print("=" * 60)
    
    # Validate API key
    missing = config.validate()
    if missing:
        print("\nERROR: Missing required configuration:")
        for key in missing:
            print(f"  - {key}")
        print("\nPlease create a .env file with your API key:")
        print("  cp .env.example .env")
        print("  # Then edit .env and add your MDC_API_KEY")
        return
    
    output_dir = config.mozilla_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Use temporary directory for downloads and extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download and process train/dev data
        print("\n" + "=" * 60)
        print("Downloading Train/Dev Data")
        print("=" * 60)
        
        try:
            print("Getting download URL from MDC API...")
            train_url = get_download_url(config.mdc_train_dev_dataset_id, config.mdc_api_key)
            print("  Download URL obtained")
            
            train_tarball = temp_path / "train_dev.tar.gz"
            download_file(train_url, train_tarball, desc="Downloading train/dev")
            
            train_extract_dir = temp_path / "train_dev_extracted"
            extract_tarball(train_tarball, train_extract_dir)
            
            organize_train_validation_data(train_extract_dir, output_dir)
            
        except requests.HTTPError as e:
            print(f"ERROR: API request failed - {e}")
            print("Check that your MDC_API_KEY is valid")
            return
        except Exception as e:
            print(f"ERROR: {e}")
            return
    
    # Verify counts (train/val only; test data not downloaded)
    verify_counts(output_dir)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nData saved to: {output_dir}")
    print(f"  Languages: {len(LANGUAGES)}")
    print(f"  Train/Val audios: {output_dir / 'shared_train_validation_audios'}")


if __name__ == "__main__":
    main()
