# dataset='rgbd_dataset_freiburg1_desk'
# dataset='rgbd_dataset_freiburg2_xyz'
dataset='rgbd_dataset_freiburg3_long_office_household'

gt="/home/tuandang/workspace/datasets/tum/extract/${dataset}/groundtruth.txt"
est="/home/tuandang/workspace/tnslam/log/tum/${dataset}/est.csv"

echo "Ground Truth: ${gt}"
echo "Estimated   : ${est}"
python evaluate_ate.py $gt $est --verbose