// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-bc -o - %s | opt -std-compile-opts | llvm-dis > %t
// RUN: grep "ret i32" %t | count 1
// RUN: grep "ret i32 10" %t | count 1

// Ensure that default after a case range is not ignored.

static int f1(unsigned x) {
  switch(x) {
  case 10 ... 0xFFFFFFFF:
    return 0;
  default:
    return 10;
  }
}

int g() {
  return f1(2);
}
