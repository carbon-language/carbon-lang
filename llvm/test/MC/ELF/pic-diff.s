// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r -t | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[^ ]+}}) {{[^ ]+}} {
// CHECK-NEXT:     0xC R_X86_64_PC32 baz 0x8
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:        Symbol {
// CHECK:          Name: baz (5)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }

.zero 4
.data

.zero 1
.align 4
foo:
.zero 8
.long baz - foo
