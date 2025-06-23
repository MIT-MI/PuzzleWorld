import argparse
import os
import re
import json
import pymupdf
from thefuzz import fuzz, process


def find_best_match(puzzle_name, solution_pdfs, threshold=80):
    """
    Find the best matching solution PDF for a given puzzle name using fuzzy matching.
    Returns the best-matched solution filename or None if no good match is found.
    """
    matches = process.extractOne(puzzle_name, solution_pdfs.keys(), scorer=fuzz.ratio)
    if matches and matches[1] >= threshold:
        return solution_pdfs[matches[0]]
    return None

def pdf_to_images(pdf_path, output_folder, file_prefix: str = "figure"):
    """
    Convert each page of the solution PDF into figure_{n}.png images.
    Returns a list of the image filenames (absolute paths).
    """
    figures = []
    try:
        doc = pymupdf.open(pdf_path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            if len(doc) == 1:
              figure_filename = f"{file_prefix}.png"
            else:
              figure_filename = f"{file_prefix}_{i+1}.png"
            figure_path = os.path.join(output_folder, figure_filename)
            pix.save(figure_path)
            figures.append(figure_path)
        print(f"[INFO] Generated solution images: {figures}")
    except Exception as e:
        print(f"[WARNING] Error converting {pdf_path} to images: {e}")
    return figures

def create_empty_metadata_json(title: str = ""):
    empty_metadata = {
        "title": title,
        "flavor_text": "",
        "difficulty": "",
        "solution": "",
        "reasoning": [
            {"explanation": ""},
            {"explanation": ""},
            {"explanation": ""}
        ],
        "modality": ["text"],
        "skills": [],
        "source": ""
    }

    return empty_metadata

def process_puzzle(input_dir, output_root, puzzle_pdf, solution_pdf=None):
    base_name = os.path.splitext(os.path.basename(puzzle_pdf))[0]
    puzzle_name = base_name.replace(" ", "_")
    folder_path = os.path.join(output_root, puzzle_name)
    os.makedirs(folder_path, exist_ok=True)
    
    puzzle_path = os.path.join(input_dir, puzzle_pdf)
    puzzle_metadata = create_empty_metadata_json(puzzle_name)

    # Convert puzzle PDF -> content_{n}.png
    pdf_to_images(puzzle_path, folder_path, file_prefix="content")
    
    # If there's a solution PDF, parse it
    if solution_pdf:
        solution_path = os.path.join(input_dir, solution_pdf)
        if os.path.exists(solution_path):
            # Convert each page of solution PDF into figure_{n}.png
            pdf_to_images(solution_path, folder_path)
        else:
            print(f"[INFO] No solution PDF found at {solution_path}")
    else:
        print(f"[INFO] No solution PDF matched for {puzzle_pdf}")
    
    metadata_path = os.path.join(folder_path, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(puzzle_metadata, f, indent=2)
    print(f"[INFO] Created {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Process puzzle PDFs.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing puzzle PDFs")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively process subdirectories")

    args = parser.parse_args()


    if args.recursive:
      input_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    else:
      input_dirs = [args.input_dir]

    for input_dir in input_dirs:
      output_root = f"data/puzzles/{input_dir}"
      os.makedirs(output_root, exist_ok=True)

      pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
      puzzle_pdfs = []
      solution_pdfs = {}

      # Identify puzzle vs. solution PDFs
      for pdf in pdf_files:
          if "solution" in pdf.lower():
              puzzle_candidate = re.sub(r'[-_ ]*solution.*\.pdf', '', pdf, flags=re.IGNORECASE)
              solution_pdfs[puzzle_candidate.lower()] = pdf
          else:
              puzzle_pdfs.append(pdf)

      # For each puzzle PDF, see if there's a matching solution
      for puzzle_pdf in puzzle_pdfs:
        base_name = os.path.splitext(os.path.basename(puzzle_pdf))[0]
        candidate_key = base_name.lower()
        
        # Use fuzzy matching to find the best match
        matched_solution = find_best_match(candidate_key, solution_pdfs)
        
        print(f"[INFO] Processing puzzle: {puzzle_pdf}")
        process_puzzle(input_dir, output_root, puzzle_pdf, matched_solution)

if __name__ == "__main__":
    main()
