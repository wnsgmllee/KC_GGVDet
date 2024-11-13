#!/bin/bash

# calc_DIRE.sh


folders=("Crafter" "imagenet_adm" "imagenet_glide" "imagenet_sdv5" "Lavie" "Real" "Sora" "biggan" )


for folder in "${folders[@]}"
do
  if [ -d "$folder" ]; then
   
    python3 calc_DIRE.py "$folder"
  else
    echo "Error: $folder is not a valid directory"
  fi
done

