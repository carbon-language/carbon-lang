// RUN: clang-cc -triple i386-unknown-unknown -DIRGENABLE_GNU -DIRGENABLE -fgnu-runtime -emit-llvm -o %t %s && 
// RUN: clang-cc -triple i386-unknown-unknown -DIRGENABLE_GNU -DIRGENABLE -g -fgnu-runtime -emit-llvm -o %t %s &&

// FIXME: Remove once GNU can IRgen everything.
// RUN: not clang-cc -triple i386-unknown-unknown -fgnu-runtime -emit-llvm -o %t %s

#include "objc-language-features.inc"
