// RUN: %clang_cc1 -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s

// CHECK: _Z16convert_char_rtec
char f(char x) {
  return convert_char_rte(x);
}
