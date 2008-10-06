// RUN: clang -emit-llvm %s -o %t
#define CFSTR __builtin___CFStringMakeConstantString

void f() {
  CFSTR("Hello, World!");
}

// rdar://6248329
void *G = CFSTR("yo joe");


