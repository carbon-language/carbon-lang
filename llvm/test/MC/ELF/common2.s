// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s | FileCheck %s

// Test that the common symbols are placed at the end of .bss. In this example
// it causes .bss to have size 9 instead of 8.

	.local	vimvardict
	.comm	vimvardict,1,8
	.bss
        .zero 1
	.align	8

// CHECK:        Section {
// CHECK:          Name: .bss (7)
// CHECK-NEXT:     Type:
// CHECK-NEXT:     Flags [
// CHECK:          ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 9
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
