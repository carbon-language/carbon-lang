// RUN: rm -f %t1.bc
// RUN: clang-cc -DPASS %s -emit-llvm-bc -o %t1.bc
// RUN: test -f %t1.bc
// RUN: not clang-cc %s -emit-llvm-bc -o %t1.bc
// RUN: not test -f %t1.bc

void f() {
}

#ifndef PASS
void g() {
  *10;
}
#endif
