// Verify that we can parse a simple CUDA file with or without -save-temps
// http://llvm.org/PR22936
// RUN: %clang --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:        -nocudainc -nocudalib -Werror -fsyntax-only -c %s
//
// Verify that we pass -x cuda-cpp-output to compiler after
// preprocessing a CUDA file
// RUN: %clang --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:        -Werror -### -save-temps -c %s 2>&1 | FileCheck %s
// CHECK-LABEL: "-cc1"
// CHECK: "-E"
// CHECK: "-x" "cuda"
// CHECK-LABEL: "-cc1"
// CHECK: "-x" "cuda-cpp-output"
//
// Verify that compiler accepts CUDA syntax with "-x cuda-cpp-output".
// RUN: %clang --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:        -Werror -fsyntax-only -x cuda-cpp-output -c %s

extern "C" int cudaConfigureCall(int, int);
extern "C" int __cudaPushCallConfiguration(int, int);

__attribute__((global)) void kernel() {}

void func() {
     kernel<<<1,1>>>();
}
