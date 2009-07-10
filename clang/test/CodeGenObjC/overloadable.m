// rdar://6657613
// RUN: clang-cc -emit-llvm %s -o %t &&

@class C;

// RUN: grep _Z1fP2id %t | count 1 && 
void __attribute__((overloadable)) f(id c) { }

// RUN: grep _Z1fP1C %t | count 1
void __attribute__((overloadable)) f(C *c) { }
