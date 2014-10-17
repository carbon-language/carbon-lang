// RUN: %clang_cc1 -emit-llvm-only -triple %itanium_abi_triple -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s

union A {
  int f1: 3;
  A();
};

A::A() {}

union B {
  int f1: 69;
  B();
};

B::B() {}

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:   0 | union A
// CHECK-NEXT:   0 |   int f1
// CHECK-NEXT:     | [sizeof=4, dsize=1, align=4
// CHECK-NEXT:     |  nvsize=1, nvalign=4]

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:   0 | union B
// CHECK-NEXT:   0 |   int f1
// CHECK-NEXT:     | [sizeof=16, dsize=9, align=8
// CHECK-NEXT:     |  nvsize=9, nvalign=8]

