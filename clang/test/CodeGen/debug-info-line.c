// RUN: %clang_cc1 -w -gline-tables-only -fexceptions -fcxx-exceptions -S -emit-llvm %s -o - | FileCheck %s

int f1(int a, int b) {
  // CHECK: icmp {{.*}}, !dbg [[DBG_F1:!.*]]
#line 100
  return a  //
         && //
         b;
}

// CHECK: [[DBG_F1]] = !MDLocation(line: 100,
