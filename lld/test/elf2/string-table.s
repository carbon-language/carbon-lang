// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld2 %t -o %t2
// RUN: llvm-readobj -sections -section-data %t2 | FileCheck %s
// REQUIRES: x86

.global _start
_start:

.section        foobar,"",@progbits,unique,1
.section        foobar,"T",@progbits,unique,2
.section        foobar,"",@nobits,unique,3
.section        foobar,"",@nobits,unique,4

.section bar, "a"

// Both sections are in the output and that the alloc section is first:
// CHECK:      Name: bar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:  SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x100B0

// CHECK:      Name: foobar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0

// CHECK:      Name: foobar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_TLS
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0

// CHECK:      Name: foobar
// CHECK-NEXT: Type: SHT_NOBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0

// CHECK-NOT:  Name: foobar

// Test that the string "bar" is merged into "foobar".

// CHECK:       Section {
// CHECK:       Index:
// CHECK:       Name: .shstrtab
// CHECK-NEXT:  Type: SHT_STRTAB
// CHECK-NEXT:  Flags [
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: 0x0
// CHECK-NEXT:  Offset:
// CHECK-NEXT:  Size: 51
// CHECK-NEXT:  Link: 0
// CHECK-NEXT:  Info: 0
// CHECK-NEXT:  AddressAlignment: 1
// CHECK-NEXT:  EntrySize: 0
// CHECK-NEXT:  SectionData (
// CHECK-NEXT:    0000: 002E7465 7874002E 62737300 666F6F62  |..text..bss.foob|
// CHECK-NEXT:    0010: 6172002E 73687374 72746162 002E7374  |ar..shstrtab..st|
// CHECK-NEXT:    0020: 72746162 002E7379 6D746162 002E6461  |rtab..symtab..da|
// CHECK-NEXT:    0030: 746100                               |ta.|
// CHECK-NEXT:  )
// CHECK-NEXT:}
// CHECK:        Name: .strtab
// CHECK-NEXT:   Type: SHT_STRTAB (0x3)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size: 15
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000: 005F7374 61727400 666F6F62 617200    |._start.foobar.|
// CHECK-NEXT:   )
// CHECK-NEXT: }
