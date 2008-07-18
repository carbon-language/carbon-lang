// RUN: rm -f %t1.bc 
// RUN: ! clang %s -emit-llvm-bc -o %t1.bc
// RUN: ! test -f %t1.bc

void f() {
}

void g() {
  *10;
}
