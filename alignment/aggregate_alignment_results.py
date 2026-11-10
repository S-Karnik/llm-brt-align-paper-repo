import argparse
import json
import os

def combine_json_files(input_files, output_file=None):
    combined_data = []

    for file in input_files:
        with open(file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse JSON file {file}") from e
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(combined_data, f, indent=2)
    
    return combined_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple JSON files into one.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if (f.endswith(".json") and 'time_info' not in f)]
    
    combine_json_files(input_files, args.output)
