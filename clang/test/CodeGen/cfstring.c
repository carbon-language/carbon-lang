// RUN: clang -emit-llvm %s -o %t
#define CFSTR __builtin___CFStringMakeConstantString

void f() {
  CFSTR("Hello, World!");
}
