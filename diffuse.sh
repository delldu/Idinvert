MODEL_NAME='styleganinv_ffhq256'
TARGET_LIST='examples/target.list'
CONTEXT_LIST='examples/context.list'
python diffuse.py $MODEL_NAME $TARGET_LIST $CONTEXT_LIST

