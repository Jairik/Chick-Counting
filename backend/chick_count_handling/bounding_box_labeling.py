"""
This script is an interactive reviewer for bounding-box statuses using snapshot images linked in a crossings file. Opens each image from the `snapshot_link` column, lets you tag
the row as OVERFIT or CLUSTER, and saves a new annotated crossings file.

Features:
1. Loads the crossings spreadsheet.
2. Iterates rows interactively:
   - Displays the snapshot image for the current row.
   - Overlays Excel row number, main_id, and current annotations.
3. Keyboard controls:
   - O: mark current row as overfit
   - C: mark current row as cluster
   - N / Enter / Space: next row
   - P: previous row
   - G: goto a specific Excel row (matches the spreadsheet, row 2..N)
   - Q: save & quit
4. Writes the updated table to OUTPUT_PATH.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CROSSINGS_INPUT   = ""
OUTPUT_PATH       = ""
WINDOW_NAME       = "Box Status Review"
COL_MAIN_ID       = "main_id"
COL_SNAPSHOT      = "snapshot_link"
# ───────────────────────────────────────────────────────────────────────────────

def extract_file_path_from_cell(cell_value: str) -> str:
    if not isinstance(cell_value, str):
        return ""
    s = cell_value.strip()
    if s.startswith("=") and "HYPERLINK" in s.upper():
        m = re.search(r'HYPERLINK\(\s*"([^"]+)"', s, flags=re.IGNORECASE)
        if not m:
            return ""
        url = m.group(1)
        path = url[8:] if url.lower().startswith("file:///") else url
        return path.replace("/", os.sep)
    return s

def show_image(img_path, overlay_lines):
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "FAILED TO READ IMAGE", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "IMAGE NOT FOUND", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    y = 30
    for line in overlay_lines:
        cv2.putText(img, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        y += 35
    cv2.imshow(WINDOW_NAME, img)

def ask_start_row_excel(num_data_rows: int) -> int:
    """
    Excel-style: header is row 1; first data row is row 2.
    Returns the pandas 0-based index to start at.
    """
    min_excel = 2
    max_excel = num_data_rows + 1
    print(f"Total data rows: {num_data_rows}")
    print(f"Press ENTER to start at Excel row {min_excel}, or type a starting Excel row ({min_excel}..{max_excel}).")
    s = input("> ").strip()
    if s == "":
        return 0
    try:
        excel_row = int(s)
        if excel_row < min_excel or excel_row > max_excel:
            print(f"Out of range. Starting at Excel row {min_excel}.")
            return 0
        return excel_row - 2
    except ValueError:
        print(f"Invalid input. Starting at Excel row {min_excel}.")
        return 0

def read_crossings(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise RuntimeError(f"Unsupported crossings file type: {ext}")

def write_output(df, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        raise RuntimeError(f"Unsupported output file type: {ext}")
    print(f"Output written: {path}")

def main():
    if not CROSSINGS_INPUT or not OUTPUT_PATH:
        raise RuntimeError("Set CROSSINGS_INPUT and OUTPUT_PATH.")

    df = read_crossings(CROSSINGS_INPUT)

    cols_lower = {c.lower(): c for c in df.columns}
    if COL_MAIN_ID.lower() not in cols_lower or COL_SNAPSHOT.lower() not in cols_lower:
        raise RuntimeError(f"Crossings file must include '{COL_MAIN_ID}' and '{COL_SNAPSHOT}' columns.")
    main_id_col  = cols_lower[COL_MAIN_ID.lower()]
    snap_col     = cols_lower[COL_SNAPSHOT.lower()]

    if "is_overfit" not in df.columns: df["is_overfit"] = ""
    if "is_cluster" not in df.columns: df["is_cluster"] = ""

    total = len(df)
    idx = ask_start_row_excel(total)

    print("\nControls:")
    print("  O = mark current row as overfit (sets is_overfit='overfit')")
    print("  C = mark current row as cluster (sets is_cluster='cluster')")
    print("  N / Enter / Space = next row")
    print("  P = previous row")
    print("  G = goto specific Excel row (2..{})".format(total+1))
    print("  Q = save & quit\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 800)

    while True:
        if idx < 0: idx = 0
        if idx >= total:
            print("Reached end.")
            break

        row = df.iloc[idx]
        img_path = extract_file_path_from_cell(row[snap_col])

        overlay = [
            f"Excel row: {idx+2}/{total+1}", 
            f"main_id: {row[main_id_col]}",
            f"is_overfit: {row.get('is_overfit','')}   is_cluster: {row.get('is_cluster','')}",
            "Keys: O=overfit, C=cluster, N/Enter/Space=next, P=prev, G=goto (Excel row), Q=quit"
        ]
        show_image(img_path, overlay)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('o'), ord('O')):
            df.at[df.index[idx], "is_overfit"] = "overfit"
            print(f"Excel row {idx+2}: is_overfit = overfit")
            idx += 1
        elif key in (ord('c'), ord('C')):
            df.at[df.index[idx], "is_cluster"] = "cluster"
            print(f"Excel row {idx+2}: is_cluster = cluster")
            idx += 1
        elif key in (ord('p'), ord('P')):
            idx -= 1
        elif key in (ord('g'), ord('G')):
            cv2.destroyWindow(WINDOW_NAME)
            try:
                min_excel = 2
                max_excel = total + 1
                excel_row = int(input(f"Goto Excel row ({min_excel}..{max_excel}): ").strip())
                if min_excel <= excel_row <= max_excel:
                    idx = excel_row - 2
                else:
                    print("Out of range; staying put.")
            except ValueError:
                print("Invalid number; staying put.")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1200, 800)
        elif key in (ord('n'), ord('N'), 13, 32):
            idx += 1
        else:
            print("Use O / C / N / P / G / Q.")

    cv2.destroyAllWindows()
    write_output(df, OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
