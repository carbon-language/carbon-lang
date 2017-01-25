// RUN: %clang -E -o %t.i %s
// RUN: %clang -c -o %t.o %t.i
// RUN: llvm-objdump -t %t.o | FileCheck %s
// CHECK: l{{ +}}df{{ +}}*ABS*{{ +}}{{0+}}{{.+}}preprocessed-input.c{{$}}
