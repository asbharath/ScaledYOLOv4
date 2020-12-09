FROM nvcr.io/nvidia/pytorch:20.06-py3
# FROM pytorch/pytorch:latest # Need to investigate why this doesn't work

# Mish is an activation function used in the csp
RUN apt-get update && \
cd / && \
git clone https://github.com/thomasbrandon/mish-cuda && \
cd mish-cuda && \
mv external/CUDAApplyUtils.cuh csrc && \
python setup.py build install
