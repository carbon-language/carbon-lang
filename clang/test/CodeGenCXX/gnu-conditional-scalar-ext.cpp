// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// rdar: // 8353567
// pr7726

extern "C" int printf(...);

int main(int argc, char **argv) {
// CHECK: phi i8* [ inttoptr (i64 3735928559 to i8*),
    printf("%p\n", (void *)0xdeadbeef ? : (void *)0xaaaaaa);
    return 0;
}

// rdar://8446940
namespace radar8446940 {
extern "C" void abort();

int main () {
  char x[1];
  char *y = x ? : 0;

  if (x != y)
    abort();
}
}

namespace radar8453812 {
extern "C" void abort();
_Complex int getComplex(_Complex int val) {
  static int count;
  if (count++)
    abort();
  return val;
}

_Complex int cmplx() {
    _Complex int cond;
    _Complex int rhs;

    return getComplex(1+2i) ? : rhs;
}

// lvalue test
void foo (int& lv) {
  ++lv;
}

int global = 1;

int &cond() {
  static int count;
  if (count++)
    abort();
  return global;
}


int main() {
  cmplx();
  int rhs = 10;
  foo (cond()? : rhs);
  return  global-2;
}
}
