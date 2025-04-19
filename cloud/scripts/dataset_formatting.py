from typing import List, Tuple
import os
from pathlib import Path
import argparse

def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing the images to process."
    )
    parser.add_argument(
        "--A", type=str, required=True,
        help="Directory containing the images of modality A to process."
    )
    parser.add_argument(
        "--B", type=str, required=True,
        help="Directory containing the images of modality B to process."
    )
    parser.add_argument(
        "--folders", type=str, nargs=4, required=True,
        help="List of folder names to create for training and testing data."
    )
    parser.add_argument(
        "--alpha", type=float, required=True,
        help="Ratio of training data to total data."
    )
    return parser


def browse_folder(
        path: Path, 
        A: str, 
        B: str) -> Tuple[List[Path], List[Path]]:
    """
    Browse a directory and return a list of all the png/jpg files in it and its subfolder.
    """
    A_path = path / A
    B_path = path / B
    A_files = sorted([p for p in A_path.glob("**/*") if p.suffix in [".jpg", ".png"] and p.is_file()])
    B_files = sorted([p for p in B_path.glob("**/*") if p.suffix in [".jpg", ".png"] and p.is_file()])
    return A_files, B_files


def check_existence(
        filenames: List[Path]) -> None:
    """
    Check if the paths exist.
    Args:
        filenames (List[Path]): List of filenames.
    Returns:
        None
    """
    for f in filenames:
        if not f.exists():
            raise FileNotFoundError(f"{f} does not exist")


def create_folders(
        input_dir: Path, 
        folder_list: List[str]) -> None:
    """
    Create folders for training and testing data.
    Args:
        input_dir (Path): Input directory.
        folder_list (List[str]): List of folder names to create.
    Returns:
        None
    """
    for folder in folder_list:
        path = input_dir / folder
        path.mkdir(parents=True, exist_ok=True)


def split_train_test(
        filenames: List[Path], 
        alpha: float) -> Tuple[List[Path], List[Path]]:
    """
    Split the filenames into training and testing sets.
    Args:
        filenames (List[Path]): List of filenames.
        alpha (float): Ratio of training data to total data.
    Returns:
        Tuple[List[Path], List[Path]]: Training and testing filenames.
    """
    total = len(filenames)
    split_idx = int(total * alpha)
    train = filenames[:split_idx]
    test = filenames[split_idx:]
    return train, test


def create_symlinks(
        filenames: List[Path],
        input_dir: Path,
        split_dir: Path) -> None:
    """
    Create symbolic links for the training and testing images.
    Args:
        filenames (List[Path]): List of filenames.
        input_dir (Path): Input directory.
        split_dir (Path): Split directory.
    Returns:
        None
    """
    for f in filenames:
        dst = split_dir / f.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(f.resolve(), dst)


def process(input_dir: str, A: str, B: str, folders: List[str], alpha: float) -> None:
    """
    Format the dataset for training and testing.
    Args:
        data_dir (str): Directory containing the images.
        A (str): Name of the first dataset.
        B (str): Name of the second dataset.
        folders (List[str]): List of folder names to create.
        alpha (float): Ratio of training data to total data.
    Returns:
        None
    """
    input_path = Path(input_dir)
    datasets_dir = input_path / "pytorch-CycleGAN-and-pix2pix" / "datasets" / f"{A}2{B}"

    A_files, B_files = browse_folder(input_path, A, B)
    check_existence(A_files + B_files)

    create_folders(datasets_dir, folders)

    A_train, A_test = split_train_test(A_files, alpha)
    B_train, B_test = split_train_test(B_files, alpha)

    folder_map = {
        folders[0]: A_train,
        folders[1]: B_train,
        folders[2]: A_test,
        folders[3]: B_test
    }

    for folder, files in folder_map.items():
        target_dir = datasets_dir / folder
        create_symlinks(files, input_path, target_dir)


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()

    process(args.input_dir, args.A, args.B, args.folders, args.alpha)


if __name__ == "__main__":
    main()
