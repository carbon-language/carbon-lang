// RUN: %clang_cc1 -std=c++0x -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

int* a = nullptr;

void f() {
  int* a = nullptr;
}

typedef decltype(nullptr) nullptr_t;

nullptr_t get_nullptr();

struct X { };
void g() {
  // CHECK: call i8* @_Z11get_nullptrv()
  int (X::*pmf)(int) = get_nullptr();
}
