// RUN: clang -emit-llvm %s
#define CFSTR __builtin___CFStringMakeConstantString

void f() {
  CFSTR("Hello, World!");
}