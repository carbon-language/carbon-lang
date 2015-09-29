// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: lld -flavor gnu2 -shared %t2.o -o %t2.so
// RUN: lld -flavor gnu2 %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s
// REQUIRES: x86

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x16000
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.dyn {
// CHECK-NEXT:     0x15000 R_386_GLOB_DAT bar 0x0
// CHECK-NEXT:     0x15004 R_386_GLOB_DAT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 0x16000 - (0x11000 + 1) - 4 = 20475
// 0x16000 - (0x11005 + 1) - 4 = 20470
// 0x16008 - (0x1100a + 1) - 4 = 20473

// DISASM:      _start:
// DISASM-NEXT:   11000:  e9 fb 4f 00 00  jmp  20475
// DISASM-NEXT:   11005:  e9 f6 4f 00 00  jmp  20470
// DISASM-NEXT:   1100a:  e9 f9 4f 00 00  jmp  20473

// 0x15000 = 86016
// 0x15004 = 86020

// DISASM:      Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   16000:  ff 25 00 50 01 00  jmpl *86016
// DISASM-NEXT:   16006:  90                 nop
// DISASM-NEXT:   16007:  90                 nop
// DISASM-NEXT:   16008:  ff 25 04 50 01 00  jmpl *86020
// DISASM-NEXT:   1600e:  90                 nop
// DISASM-NEXT:   1600f:  90                 nop

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
