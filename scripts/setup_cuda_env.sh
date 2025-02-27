CUDA_VERSION=cuda-12.6
export CPATH=/usr/local/${CUDA_VERSION}/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/${CUDA_VERSION}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/${CUDA_VERSION}/bin:$PATH

echo $CPATH
echo $LD_LIBRARY_PATH
echo $PATH