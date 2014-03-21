// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -t - | FileCheck %s

// Test that a variable declared with "var = other_var + cst" is in the same
// section as other_var and its value is the value of other_var + cst.

sym_a:
sym_d = sym_a + 1


// CHECK:       Symbol {
// CHECK:         Name: sym_a
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text (0x1)
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: sym_d
// CHECK-NEXT:    Value: 0x1
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local (0x0)
// CHECK-NEXT:    Type: None (0x0)
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section:  (0xFFF1)
// CHECK-NEXT:  }
