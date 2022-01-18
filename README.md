# cutlass

see the origin README from NVIDIA [here](https://github.com/liujuanLT/cutlass/blob/master/README_NVIDIA.md)

## test on V100
``` shell
    export PATH=/usr/local/cuda-11.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
    export CUDACXX=/usr/local/cuda-11.2/bin/nvcc

    mkdir build && cd build
    cmake .. -DCUTLASS_NVCC_ARCHS=70
    cd build/examples/07_volta_tensorop_gemm/
    make -j4
    ./07_volta_tensorop_gemm  --m=1024 --k=1024 --n=1024 --niters=1000
```

## test on A10
``` shell
  export PATH=/usr/local/cuda-11.3/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
  export CUDACXX=/usr/local/cuda-11.3/bin/nvcc
  mkdir build && cd build
  cmake .. -DCUTLASS_NVCC_ARCHS=86
  cd build/examples/14_ampere_tf32_tensorop_gemm
  make -j4
  ./14_ampere_tf32_tensorop_gemm
```