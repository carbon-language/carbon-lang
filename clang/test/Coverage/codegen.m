// RUN: clang -fnext-runtime -emit-llvm -o %t %s
// RUN: clang -fnext-runtime -emit-llvm-bc -o %t %s
// XFAIL

#include "objc-language-features.inc"
