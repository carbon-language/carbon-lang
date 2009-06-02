// RUN: clang-cc -emit-llvm -o %t %s &&
// RUN: grep '@f0' %t | count 0 &&
// RUN: clang-cc -disable-llvm-optzns -emit-llvm -o %t %s &&
// RUN: grep '@f0' %t | count 2

//static int f0() { 
static int __attribute__((always_inline)) f0() { 
  return 1;
}

int f1() {
  return f0();
}
