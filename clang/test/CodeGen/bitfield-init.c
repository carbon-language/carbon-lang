// RUN: clang %s -emit-llvm 
typedef struct { unsigned int i: 1; } c;
const c d = { 1 };

