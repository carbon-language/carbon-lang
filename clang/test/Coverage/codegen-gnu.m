// RUN: clang -DIRGENABLE_GNU -DIRGENABLE -fgnu-runtime -emit-llvm -o %t %s && 
// RUN: clang -DIRGENABLE_GNU -DIRGENABLE -g -fgnu-runtime -emit-llvm -o %t %s &&

// FIXME: Remove once GNU can IRgen everything.
// RUN: ! clang -fgnu-runtime -emit-llvm -o %t %s

#include "objc-language-features.inc"
