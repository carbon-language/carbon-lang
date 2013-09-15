// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -t | FileCheck %s

// Test that we produce the group sections and that they are a the beginning
// of the file.

// CHECK:        Section {
// CHECK:          Index: 1
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Type: SHT_GROUP
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 12
// CHECK-NEXT:     Link: 13
// CHECK-NEXT:     Info: 1
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 4
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 2
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Type: SHT_GROUP
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x4C
// CHECK-NEXT:     Size: 8
// CHECK-NEXT:     Link: 13
// CHECK-NEXT:     Info: 2
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 4
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index: 3
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Type: SHT_GROUP
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x54
// CHECK-NEXT:     Size: 12
// CHECK-NEXT:     Link: 13
// CHECK-NEXT:     Info: 13
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 4
// CHECK-NEXT:   }

// Test that g1 and g2 are local, but g3 is an undefined global.

// CHECK:        Symbol {
// CHECK:          Name: g1 (1)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .foo (0x7)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: g2 (4)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .group (0x2)
// CHECK-NEXT:   }

// CHECK:        Symbol {
// CHECK:          Name: g3 (7)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: (0x0)
// CHECK-NEXT:   }


	.section	.foo,"axG",@progbits,g1,comdat
g1:
        nop

        .section	.bar,"ax?",@progbits
        nop

        .section	.zed,"axG",@progbits,g2,comdat
        nop

        .section	.baz,"axG",@progbits,g3,comdat
        .long g3
