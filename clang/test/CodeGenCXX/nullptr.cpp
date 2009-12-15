// RUN: %clang_cc1 -std=c++0x  %s -emit-llvm -o %t

int* a = nullptr;

void f() {
  int* a = nullptr;
}
