// RUN: rm -f %t1.bc
// RUN: %clang_cc1 -DPASS %s -emit-llvm-bc -o %t1.bc
// RUN: test -f %t1.bc
// RUN: rm -f %t1.bc
// RUN: not %clang_cc1 %s -emit-llvm-bc -o %t1.bc
// RUN: not test -f %t1.bc

void f() {
}

#ifndef PASS
void g() {
  *10;
}
#endif
