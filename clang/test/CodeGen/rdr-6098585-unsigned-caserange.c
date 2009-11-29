// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm-bc -o - %s | opt -std-compile-opts | llvm-dis > %t
// RUN: grep "ret i32" %t | count 1
// RUN: grep "ret i32 3" %t | count 1

int f2(unsigned x) {
  switch(x) {
  default:
    return 3;
  case 0xFFFFFFFF ... 1: // This range should be empty because x is unsigned.
    return 0;
  }
}
