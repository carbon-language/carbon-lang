// RUN: clang-cc -fnext-runtime -emit-llvm -o %t %s
// RUN: clang-cc -g -fnext-runtime -emit-llvm -o %t %s

#include "objc-language-features.inc"
