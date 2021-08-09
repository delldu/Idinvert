MODEL_NAME='styleganinv_ffhq256'
IMAGE_DIR='results/inversion/test'

BOUNDARY='boundaries/stylegan_ffhq256/expression.npy'
python manipulate.py $MODEL_NAME $IMAGE_DIR $BOUNDARY

BOUNDARY='boundaries/stylegan_ffhq256/age.npy'
python manipulate.py $MODEL_NAME $IMAGE_DIR $BOUNDARY

BOUNDARY='boundaries/stylegan_ffhq256/eyeglasses.npy'
python manipulate.py $MODEL_NAME $IMAGE_DIR $BOUNDARY

BOUNDARY='boundaries/stylegan_ffhq256/gender.npy'
python manipulate.py $MODEL_NAME $IMAGE_DIR $BOUNDARY

BOUNDARY='boundaries/stylegan_ffhq256/pose.npy'
python manipulate.py $MODEL_NAME $IMAGE_DIR $BOUNDARY

