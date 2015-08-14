// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-readobj -symbols %t2 | FileCheck %s
// REQUIRES: x86

.type _start, @function
.globl _start
_start:

.type foo, @object
.weak foo
foo:

.type bar, @object
.weak bar
.long bar

// CHECK:      Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name:  (0)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local (0x0)
// CHECK-NEXT:     Type: None (0x0)
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: _start
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global (0x1)
// CHECK-NEXT:     Type: Function
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Weak (0x2)
// CHECK-NEXT:     Type: Object (0x1)
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: foo
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Weak (0x2)
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT: ]
