// RUN: clang -cc1 -emit-llvm -o %t %s
// RUN: clang -cc1 -g -emit-llvm -o %t %s

#include "objc-language-features.inc"
