// RUN: %clang_cc1 -emit-llvm -o - %s
struct S { 
  static int i;
};

void f() { 
  int a = S::i;
}
