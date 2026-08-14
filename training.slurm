#!/bin/bash
#SBATCH --job-name=yield-cornbelt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sridhar.86@buckeyemail.osu.edu
#SBATCH --time=08:00:00
#SBATCH --account=PAS2699
#SBATCH --output=yield-training.%j.out
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1

module load python/3.12

source activate yield_hf

python scripts/train_hf.py \
  --train_file data/cornbelt/train.h5 \
  --val_file data/cornbelt/val.h5 \
  --test_file data/cornbelt/test.h5 \
  --weather_vars prcp,srad,swe,tmax,tmin,vp \
  --soil_vars bdod_mean_0-5cm,bdod_mean_5-15cm,bdod_mean_15-30cm,bdod_mean_30-60cm,bdod_mean_60-100cm,bdod_mean_100-200cm,cec_mean_0-5cm,cec_mean_5-15cm,cec_mean_15-30cm,cec_mean_30-60cm,cec_mean_60-100cm,cec_mean_100-200cm,cfvo_mean_0-5cm,cfvo_mean_5-15cm,cfvo_mean_15-30cm,cfvo_mean_30-60cm,cfvo_mean_60-100cm,cfvo_mean_100-200cm,clay_mean_0-5cm,clay_mean_5-15cm,clay_mean_15-30cm,clay_mean_30-60cm,clay_mean_60-100cm,clay_mean_100-200cm,nitrogen_mean_0-5cm,nitrogen_mean_5-15cm,nitrogen_mean_15-30cm,nitrogen_mean_30-60cm,nitrogen_mean_60-100cm,nitrogen_mean_100-200cm,ocd_mean_0-5cm,ocd_mean_5-15cm,ocd_mean_15-30cm,ocd_mean_30-60cm,ocd_mean_60-100cm,ocd_mean_100-200cm,ocs_mean_0-5cm,ocs_mean_5-15cm,ocs_mean_15-30cm,ocs_mean_30-60cm,ocs_mean_60-100cm,ocs_mean_100-200cm,phh2o_mean_0-5cm,phh2o_mean_5-15cm,phh2o_mean_15-30cm,phh2o_mean_30-60cm,phh2o_mean_60-100cm,phh2o_mean_100-200cm,sand_mean_0-5cm,sand_mean_5-15cm,sand_mean_15-30cm,sand_mean_30-60cm,sand_mean_60-100cm,sand_mean_100-200cm,silt_mean_0-5cm,silt_mean_5-15cm,silt_mean_15-30cm,silt_mean_30-60cm,silt_mean_60-100cm,silt_mean_100-200cm,soc_mean_0-5cm,soc_mean_5-15cm,soc_mean_15-30cm,soc_mean_30-60cm,soc_mean_60-100cm,soc_mean_100-200cm \
  --crop corn \
  --time_agg weekly \
  --train_cutoffs 20,24,28,32,36,40,44,48,52 \
  --eval_cutoffs 20,24,28,32,36,40,44,48,52 \
  --epochs 30 \
  --lr 3e-5 \
  --batch_size 32 \
  --out_dir checkpoints