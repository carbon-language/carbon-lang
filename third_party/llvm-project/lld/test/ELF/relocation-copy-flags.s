// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/relocation-copy.s -o %t2.o
// RUN: ld.lld %t2.o -o %t2.so -shared -soname=so
// RUN: ld.lld --hash-style=sysv %t.o %t2.so -o %t.exe
// RUN: llvm-readobj -S --section-data -r %t.exe | FileCheck %s

// Copy relocate x in a non-writable position.
        .global _start
_start:
        .quad x

// Resolved to 0 in a non-alloc section.
        .section foo
        .quad y

// Produce a dynamic relocation in a writable position.
        .section bar, "aw"
        .quad z

// CHECK:      Name: .text
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 90332000
// CHECK-NEXT: )

// CHECK:      Name: bar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x203380
// CHECK-NEXT: Offset: 0x380
// CHECK-NEXT: Size: 8
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 1
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// CHECK-NEXT:   0000: 00000000
// CHECK-NEXT: )

// CHECK:      Name: foo
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0
// CHECK-NEXT: Offset: 0x388
// CHECK-NEXT: Size: 8
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 1
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// CHECK-NEXT:   0000: 00000000
// CHECK-NEXT: )

// CHECK:      Relocations [
// CHECK-NEXT:   Section (4) .rela.dyn {
// CHECK-NEXT:     0x203390 R_X86_64_COPY x 0x0
// CHECK-NEXT:     0x203380 R_X86_64_64 z 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
