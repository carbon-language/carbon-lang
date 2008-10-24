// RUN: clang -DIRGENABLE -fnext-runtime -emit-llvm -o %t %s &&
// RUN: clang -DIRGENABLE -g -fnext-runtime -emit-llvm -o %t %s &&

// FIXME: Remove IRGENABLE when possible.
// RUN: ! clang -fnext-runtime -emit-llvm -o %t %s

#include "objc-language-features.inc"
