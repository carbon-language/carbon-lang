// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: lld -flavor gnu2 %t -o %t2
// RUN: llvm-readobj -sections -section-data %t2 | FileCheck %s
// REQUIRES: x86

.global _start
_start:

.global hidden
.hidden hidden
hidden:

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
// CHECK-NEXT: Address: 0x1000

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

// Test that the sting "bar" is merged into "foobar" and that we don't output
// the name of a hidden symbol.

// CHECK:      Section {
// CHECK:        Index:
// CHECK:        Name: .strtab
// CHECK-NEXT:   Type: SHT_STRTAB (0x3)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size: 48
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000: 002E7465 7874005F 73746172 74002E62  |..text._start..b|
// CHECK-NEXT:     0010: 73730066 6F6F6261 72002E73 74727461  |ss.foobar..strta|
// CHECK-NEXT:     0020: 62002E73 796D7461 62002E64 61746100  |b..symtab..data.|
// CHECK-NEXT:   )
// CHECK-NEXT: }
