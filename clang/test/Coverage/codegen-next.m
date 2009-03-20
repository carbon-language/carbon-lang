// RUN: clang -fnext-runtime -emit-llvm -o %t %s &&
// RUN: clang -g -fnext-runtime -emit-llvm -o %t %s

#include "objc-language-features.inc"
