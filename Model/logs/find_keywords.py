import os

def find_keywords(root_dir='.'):
    """
    Recursively search for .err files containing the keyword 'optuna' in a directory.
    """
    keyword = "checkpoints_old_isi"
    found_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.out'):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            if keyword in line:
                                found_files.append(fpath)
                                break  # Found keyword, skip rest of file
                except Exception as e:
                    print(f"⚠️ Error reading {fpath}: {e}")

    print(f"\nFound {len(found_files)} file(s) containing '{keyword}':")
    for f in found_files:
        print(f)

if __name__ == "__main__":
    find_keywords('.')  # Replace '.' with your directory path if needed

