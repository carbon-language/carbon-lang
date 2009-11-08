// RUN: rm -f %t1.bc
// RUN: not clang-cc %s -emit-llvm-bc -o %t1.bc
// RUN: not test -f %t1.bc

void f() {
}

void g() {
  *10;
}
