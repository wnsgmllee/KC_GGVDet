#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

FOLDER_PATH="$1"

# Check if the provided argument is a directory
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: $FOLDER_PATH is not a valid directory."
    exit 1
fi

# Run the Python script
python3 to_freq.py "$FOLDER_PATH"

