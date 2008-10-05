// RUN: clang -emit-llvm -o %t %s &&
// RUN: clang -emit-llvm-bc -o %t %s &&
// RUN: clang -g -emit-llvm-bc -o %t %s

#include "c-language-features.inc"
