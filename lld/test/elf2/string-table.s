// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-readobj -sections -section-data %t2 | FileCheck %s
// REQUIRES: x86

.global _start
_start:

.section bar
.section foobar

// Both sections are in the output:
// CHECK: Name: bar
// CHECK: Name: foobar

// Test that the sting "bar" is merged into "foobar"

// CHECK:      Section {
// CHECK:        Index: 6
// CHECK-NEXT:   Name: .strtab
// CHECK-NEXT:   Type: SHT_STRTAB (0x3)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size: 33
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000: 002E7465 7874002E 62737300 666F6F62  |..text..bss.foob|
// CHECK-NEXT:     0010: 6172002E 73747274 6162002E 64617461  |ar..strtab..data|
// CHECK-NEXT:     0020: 00                                   |.|
// CHECK-NEXT:   )
// CHECK-NEXT: }
