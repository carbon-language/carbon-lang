// RUN: llvm-mc -triple i386-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj -t | FileCheck %s

.lcomm A, 5
.lcomm B, 32 << 20

// CHECK:        Symbol {
// CHECK:          Name: A (1)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 5
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .bss (0x3)
// CHECK-NEXT:   }
// CHECK:        Symbol {
// CHECK:          Name: B (3)
// CHECK-NEXT:     Value: 0x5
// CHECK-NEXT:     Size: 33554432
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .bss (0x3)
// CHECK-NEXT:   }
