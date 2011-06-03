// REQUIRES: x86-registered-target
// RUN: %clang -ccc-host-triple i386-apple-darwin10 -S -g -dA %s -o - | FileCheck %s
int global;
// CHECK: ascii   "localstatic"          ## DW_AT_name
// CHECK: asciz   "global" ## External Name
int main() { 
  static int localstatic;
  return 0;
}
