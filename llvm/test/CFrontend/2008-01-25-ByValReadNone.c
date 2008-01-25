// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | grep readonly
// RUN: %llvmgcc -O3 -S -o - -emit-llvm %s | not grep readnone


// The struct being passed byval means that we need to mark the 
// function readonly instead of readnone.  Readnone would allow
// stores to the arg to be deleted in the caller.
struct S { int A[1000]; };
int __attribute__ ((const)) f(struct S x) { return x.A[0]; }

