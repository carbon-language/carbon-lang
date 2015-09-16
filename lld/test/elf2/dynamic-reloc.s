// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: lld -flavor gnu2 -shared %t2.o -o %t2.so
// RUN: lld -flavor gnu2 %t.o %t2.so -o %t
// RUN: llvm-readobj -r --expand-relocs -s %t | FileCheck %s
// REQUIRES: x86

// CHECK:      Name: .text
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: [[ADDR:.*]]

// CHECK:      Index: 4
// CHECK-NEXT: Name: .dynsym

// CHECK:      Name: .rela.dyn
// CHECK-NEXT: Type: SHT_RELA
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x16000
// CHECK-NEXT: Offset: 0x6000
// CHECK-NEXT: Size: 24
// CHECK-NEXT: Link: 4
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 24

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: [[ADDR]]
// CHECK-NEXT:       Type: R_X86_64_64
// CHECK-NEXT:       Symbol: bar
// CHECK-NEXT:       Addend: 0x42
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.global _start
_start:
.quad bar + 0x42
