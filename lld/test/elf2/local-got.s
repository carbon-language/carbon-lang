// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: lld -flavor gnu2 %t.o -o %t
// RUN: llvm-readobj -s -r -section-data %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s

        .globl _start
_start:
	call foo@gotpcrel

        .global foo
foo:
        nop

// 0x12000 - 0x11000 - 5 = 4091
// DISASM:      _start:
// DISASM-NEXT:   11000: {{.*}} callq 4091

// DISASM:      foo:
// DISASM-NEXT:   11005: {{.*}} nop

// CHECK:      Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x12000
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 8
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// 0x11005 in little endian
// CHECK-NEXT:   0000: 05100100 00000000                    |........|
// CHECK-NEXT: )

// CHECK:      Relocations [
// CHECK-NEXT: ]
