// RUN: clang-cc %s -emit-llvm -o %t

int* a = __null;
int b = __null;

void f() {
  int* c = __null;
  int d = __null;
}
