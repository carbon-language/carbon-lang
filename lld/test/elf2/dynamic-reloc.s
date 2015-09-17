// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: lld -flavor gnu2 -shared %t2.o -o %t2.so
// RUN: lld -flavor gnu2 %t.o %t2.so -o %t
// RUN: llvm-readobj -dynamic-table -r --expand-relocs -s %t | FileCheck %s
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
// CHECK-NEXT: Address: [[RELAADDR:.*]]
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: [[RELASIZE:.*]]
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

// CHECK: DynamicSection [
// CHECK-NEXT:  Tag                Type                 Name/Value
// CHECK-NEXT:  0x0000000000000007 RELA                 [[RELAADDR]]
// CHECK-NEXT:  0x0000000000000008 RELASZ               [[RELASIZE]] (bytes)
// CHECK-NEXT:  0x0000000000000006 SYMTAB
// CHECK-NEXT:  0x0000000000000005 STRTAB
// CHECK-NEXT:  0x000000000000000A STRSZ
// CHECK-NEXT:  0x0000000000000004 HASH
// CHECK-NEXT:  0x0000000000000001 NEEDED               SharedLibrary ({{.*}}2.so)
// CHECK-NEXT:  0x0000000000000000 NULL                 0x0
// CHECK-NEXT: ]


.global _start
_start:
.quad bar + 0x42
