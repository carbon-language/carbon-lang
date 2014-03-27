// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -t - | FileCheck %s

// Test that a variable declared with "var = other_var + cst" is in the same
// section as other_var and its value is the value of other_var + cst.

        .data
        .globl	sym_a
        .size sym_a, 42
        .byte 42
        .type sym_a, @object
sym_a:

// CHECK:       Symbol {
// CHECK:         Name: sym_a
// CHECK-NEXT:    Value: 0x1
// CHECK-NEXT:    Size: 42
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: Object
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .data
// CHECK-NEXT:  }

        .long 42
        .globl sym_b
sym_b:
        .globl sym_c
sym_c = sym_a
// CHECK:       Symbol {
// CHECK:         Name: sym_c
// CHECK-NEXT:    Value: 0x1
// CHECK-NEXT:    Size: 42
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: Object
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .data
// CHECK-NEXT:  }

        .globl sym_d
sym_d = sym_a + 1
// CHECK:       Symbol {
// CHECK:         Name: sym_d
// CHECK-NEXT:    Value: 0x2
// CHECK-NEXT:    Size: 42
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: Object
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .data
// CHECK-NEXT:  }

        .globl sym_e
sym_e = sym_a + (sym_b - sym_a) * 3
// CHECK:       Symbol {
// CHECK:         Name: sym_e
// CHECK-NEXT:    Value: 0xD
// CHECK-NEXT:    Size: 42
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: Object
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .data
// CHECK-NEXT:  }


        .globl sym_f
sym_f = sym_a + (1 - 1)
// CHECK:       Symbol {
// CHECK:         Name: sym_f
// CHECK-NEXT:    Value: 0x1
// CHECK-NEXT:    Size: 42
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: Object
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .data
// CHECK-NEXT:  }
