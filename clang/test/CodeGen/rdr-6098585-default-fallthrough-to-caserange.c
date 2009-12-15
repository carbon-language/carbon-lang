// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-bc -o - %s | opt -std-compile-opts | llvm-dis > %t
// RUN: grep "ret i32 10" %t

// Ensure that this doesn't compile to infinite loop in g() due to
// miscompilation of fallthrough from default to a (tested) case
// range.

static int f0(unsigned x) {
  switch(x) {
  default:
    x += 1;
  case 10 ... 0xFFFFFFFF:
    return 0;
  }
}

int g() {
  f0(1);
  return 10;
}
