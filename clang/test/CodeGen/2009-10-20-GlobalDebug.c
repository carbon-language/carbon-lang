// RUN: %clang -ccc-host-triple i386-apple-darwin10 -S -g -dA %s -o - | FileCheck %s
int global;
// CHECK: asciz   "global" ## External Name
// CHECK: asciz   "localstatic"          ## External Name
int main() { 
  static int localstatic;
  return 0;
}
