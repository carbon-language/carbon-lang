// RUN: clang-cc -emit-llvm < %s | grep i1 | count 1

// Check that the type of this global isn't i1
_Bool test = &test;
