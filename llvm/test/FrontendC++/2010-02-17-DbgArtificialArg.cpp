// RUN: %llvmgcc -g -S %s -dA -fverbose-asm -o - | %llc -asm-verbose | FileCheck %s
// Test to artificial attribute attahed to "this" pointer type.
// Radar 7655792 and 7655002

class A {
public:
  int fn1(int i) const { return i + 2; };
};

int foo() {
  A a;
//CHECK:        .ascii   "this"                 ## DW_AT_name
//CHECK-NEXT:        .byte   0
//CHECK-NEXT:        ## DW_AT_decl_file
//CHECK-NEXT:        ## DW_AT_decl_line
//CHECK-NEXT:        ## DW_AT_type
//CHECK-NEXT:        ## DW_AT_artificial

  return a.fn1(1);
}
