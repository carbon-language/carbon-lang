// RUN: llvm-mc -triple x86_64-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -s | FileCheck %s

// This tests that when producing files for darwin9 or older we make sure
// that debug_line sections are of a minimum size to avoid the linker bug
// described in PR8715.

        .section        __DATA,__data
        .file   1 "test.c"
        .globl  _c                      ## @c
_c:
        .asciz   "hi\n"

// CHECK:       Section {
// CHECK:         Index: 2
// CHECK-NEXT:    Name: __debug_line
// CHECK-NEXT:    Segment: __DWARF
// CHECK-NEXT:    Address: 0x4
// CHECK-NEXT:    Size: 0x2C
// CHECK-NEXT:    Offset: 452
// CHECK-NEXT:    Alignment: 0
// CHECK-NEXT:    RelocationOffset: 0x0
// CHECK-NEXT:    RelocationCount: 0
// CHECK-NEXT:    Type: 0x0
// CHECK-NEXT:    Attributes [ (0x20000)
// CHECK-NEXT:      Debug (0x20000)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Reserved1: 0x0
// CHECK-NEXT:    Reserved2: 0x0
// CHECK-NEXT:  }
