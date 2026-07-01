#!/bin/sh
#SBATCH --job-name=baseline-training
#SBATCH --mail-type=ALL
#SBATCH --mail-user="sridhar.86@buckeyemail.osu.edu"
#SBATCH --time=08:00:00
#SBATCH --account=PAS2699
#SBATCH --output=output_unimodal/weekly-ye-baseline-2018.out.%j
#SBATCH --mem=100gb
#SBATCH --gpus-per-node=1

module load python/3.12
source activate yield_hf

python scripts/train_hf.py \
--train_file /fs/ess/PAS0272/sarikaa/yield_estimation_dev/data/train.h5 \
--val_file /fs/ess/PAS0272/sarikaa/yield_estimation_dev/data/val.h5 \
--test_file /fs/ess/PAS0272/sarikaa/yield_estimation_dev/data/test.h5 \
--weather_vars dayl,prcp,srad,tmax,gdd,tmin,vp,tmean,precip_3day_avg_perday,precip_7day_avg_perday,precip_14day_avg_perday \
--soil_vars ph,om,cec,awc,clay,p_ppm,k_ppm,mg_ppm,ca_ppm,k_te,mg_te,ca_te,s_ppm,zn_ppm,fe_ppm,mn_ppm,cu_ppm,b_ppm,na_ppm,sand,silt,bd,elevation,slope \
--crop corn \
--epochs 30 \
--batch_size 16 \
--out_dir outputs/intag_sare_corn_hf

