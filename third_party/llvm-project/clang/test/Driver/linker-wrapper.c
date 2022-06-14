// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX_LINK

// NVPTX_LINK: nvlink{{.*}}-m64 -o {{.*}}.out -arch sm_70 {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU_LINK

// AMDGPU_LINK: lld{{.*}}-flavor gnu --no-undefined -shared -o {{.*}}.out {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld.lld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CPU_LINK

// CPU_LINK: ld.lld{{.*}}-m elf_x86_64 -shared -Bsymbolic -o {{.*}}.out {{.*}}.o {{.*}}.o

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o
// RUN: clang-linker-wrapper --dry-run --host-triple x86_64-unknown-linux-gnu -linker-path \
// RUN:  /usr/bin/ld.lld -- -a -b -c %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HOST_LINK

// HOST_LINK: ld.lld{{.*}}-a -b -c {{.*}}.o -o a.out

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-bc.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%S/Inputs/dummy-bc.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=LTO

// LTO: ptxas{{.*}}-m64 -o {{.*}}.cubin -O2 --gpu-name sm_70 {{.*}}.s
// LTO-NOT: nvlink

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA_OMP_LINK

// CUDA_OMP_LINK: nvlink{{.*}}-m64 -o {{.*}}.out -arch sm_70 {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-obj.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.a %t-obj.o -o a.out 2>&1 | FileCheck %s --check-prefix=STATIC-LIBRARY

// STATIC-LIBRARY: nvlink{{.*}} -arch sm_70
// STATIC-LIBRARY-NOT: nvlink{{.*}} -arch sm_50

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple x86_64-unknown-linux-gnu -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

// CUDA: nvlink{{.*}}-m64 -o {{.*}}.out -arch sm_52 {{.*}}.o
// CUDA: nvlink{{.*}}-m64 -o {{.*}}.out -arch sm_70 {{.*}}.o {{.*}}.o
// CUDA: fatbinary{{.*}}-64 --create {{.*}}.fatbin --image=profile=sm_52,file={{.*}}.out --image=profile=sm_70,file={{.*}}.out

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple x86_64-unknown-linux-gnu -linker-path \
// RUN:   /usr/bin/ld --device-linker=a --device-linker=nvptx64-nvidia-cuda=b -- \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=LINKER_ARGS

// LINKER_ARGS: lld{{.*}}-flavor gnu --no-undefined -shared -o {{.*}}.out {{.*}}.o a
// LINKER_ARGS: nvlink{{.*}}-m64 -o {{.*}}.out -arch sm_70 {{.*}}.o a b
