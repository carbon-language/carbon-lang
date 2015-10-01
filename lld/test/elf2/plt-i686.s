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
// CHECK-NEXT: Address: 0x12010
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.dyn {
// CHECK-NEXT:     0x13050 R_386_GLOB_DAT bar 0x0
// CHECK-NEXT:     0x13054 R_386_GLOB_DAT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 0x12010 - (0x12000 + 1) - 4 = 11
// 0x12010 - (0x12005 + 1) - 4 = 2
// 0x12018 - (0x1200a + 1) - 4 = 9

// DISASM:      _start:
// DISASM-NEXT:   12000:  e9 0b 00 00 00  jmp  11
// DISASM-NEXT:   12005:  e9 06 00 00 00  jmp  6
// DISASM-NEXT:   1200a:  e9 09 00 00 00  jmp  9

// 0x13050 = 77904
// 0x13054 = 77908

// DISASM:      Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   12010:  ff 25 {{.*}}       jmpl *77904
// DISASM-NEXT:   12016:  90                 nop
// DISASM-NEXT:   12017:  90                 nop
// DISASM-NEXT:   12018:  ff 25 {{.*}}       jmpl *77908
// DISASM-NEXT:   1201e:  90                 nop
// DISASM-NEXT:   1201f:  90                 nop

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
