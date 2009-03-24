// RUN: clang-cc -emit-llvm < %s -o - | grep "llvm.memcpy"

char* x(char* a, char* b) {return __builtin_memcpy(a, b, 4);}
