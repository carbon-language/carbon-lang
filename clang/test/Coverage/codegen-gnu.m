// RUN: clang -fgnu-runtime -emit-llvm -o %t %s &&
// RUN: clang -g -fgnu-runtime -emit-llvm -o %t %s
// XFAIL

#include "objc-language-features.inc"
