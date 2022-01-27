// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj -S -r --symbols %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s

// CHECK:      Name: .got.plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x4020F4
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size:
// CHECK-NEXT: Link:
// CHECK-NEXT: Info:
// CHECK-NEXT: AddressAlignment:

// CHECK:      Symbol {
// CHECK:       Name: bar
// CHECK-NEXT:  Value: 0x402100
// CHECK-NEXT:  Size: 10
// CHECK-NEXT:  Binding: Global
// CHECK-NEXT:  Type: Object
// CHECK-NEXT:  Other: 0
// CHECK-NEXT:  Section: .bss
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:  Name: obj
// CHECK-NEXT:  Value: 0x40210A
// CHECK-NEXT:  Size: 10
// CHECK-NEXT:  Binding: Global
// CHECK-NEXT:  Type: Object
// CHECK-NEXT:  Other: 0
// CHECK-NEXT:  Section: .bss
// CHECK-NEXT: }

// 0x402000 - 0 = addr(.got) = 0x402000
// 0x40200A - 10 = addr(.got) = 0x402000
// 0x40200A + 5 - 15 = addr(.got) = 0x402000
// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <_start>:
// DISASM-NEXT: 4010d4: c7 81 0c 00 00 00 01 00 00 00 movl $1, 12(%ecx)
// DISASM-NEXT: 4010de: c7 81 16 00 00 00 02 00 00 00 movl $2, 22(%ecx)
// DISASM-NEXT: 4010e8: c7 81 1b 00 00 00 03 00 00 00 movl $3, 27(%ecx)

.global _start
_start:
  movl $1, bar@GOTOFF(%ecx)
  movl $2, obj@GOTOFF(%ecx)
  movl $3, obj+5@GOTOFF(%ecx)
  .type bar, @object
  .comm bar, 10
  .type obj, @object
  .comm obj, 10
