// RUN: %clang -emit-llvm -S -O0 -g %s -o - | FileCheck %s
// CHECK: !MDGlobalVariable(
unsigned char ctable1[1] = { 0001 };
