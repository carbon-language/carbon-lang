// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld2 -shared %t2.o -o %t2.so
// RUN: ld.lld2 -shared %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s
// REQUIRES: x86

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x2020
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 24
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x30A0 R_X86_64_GLOB_DAT bar 0x0
// CHECK-NEXT:     0x30A8 R_X86_64_GLOB_DAT zed 0x0
// CHECK-NEXT:     0x30B0 R_X86_64_GLOB_DAT _start 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 0x12020 - (0x12000 + 1) - 4 = 27
// 0x12020 - (0x12005 + 1) - 4 = 22
// 0x12028 - (0x1200a + 1) - 4 = 25

// DISASM:      _start:
// DISASM-NEXT:   2000:  e9 {{.*}}       jmp  27
// DISASM-NEXT:   2005:  e9 {{.*}}       jmp  22
// DISASM-NEXT:   200a:  e9 {{.*}}       jmp  25

// 0x130A0 - 0x12026  = 4218
// 0x130A8 - 0x1202e  = 4218

// DISASM:      Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   2020:  ff 25 {{.*}}       jmpq *4218(%rip)
// DISASM-NEXT:   2026:  90                 nop
// DISASM-NEXT:   2027:  90                 nop
// DISASM-NEXT:   2028:  ff 25 {{.*}}       jmpq *4218(%rip)
// DISASM-NEXT:   202e:  90                 nop
// DISASM-NEXT:   202f:  90                 nop

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
  jmp _start@plt
