// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld2 -shared %t2.o -o %t2.so
// RUN: ld.lld2 -shared %t.o %t2.so -o %t
// RUN: ld.lld2 %t.o %t2.so -o %t3
// RUN: llvm-readobj -s -r %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s
// RUN: llvm-readobj -s -r %t3 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump -d %t3 | FileCheck --check-prefix=DISASM2 %s

// REQUIRES: x86

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x1020
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 24
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x20A0 R_X86_64_GLOB_DAT bar 0x0
// CHECK-NEXT:     0x20A8 R_X86_64_GLOB_DAT zed 0x0
// CHECK-NEXT:     0x20B0 R_X86_64_GLOB_DAT _start 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK2:      Name: .plt
// CHECK2-NEXT: Type: SHT_PROGBITS
// CHECK2-NEXT: Flags [
// CHECK2-NEXT:   SHF_ALLOC
// CHECK2-NEXT:   SHF_EXECINSTR
// CHECK2-NEXT: ]
// CHECK2-NEXT: Address: 0x11020
// CHECK2-NEXT: Offset:
// CHECK2-NEXT: Size: 16
// CHECK2-NEXT: Link: 0
// CHECK2-NEXT: Info: 0
// CHECK2-NEXT: AddressAlignment: 16

// CHECK2:      Relocations [
// CHECK2-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK2-NEXT:     0x120A0 R_X86_64_GLOB_DAT bar 0x0
// CHECK2-NEXT:     0x120A8 R_X86_64_GLOB_DAT zed 0x0
// CHECK2-NEXT:   }
// CHECK2-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 0x1020 - (0x1000 + 1) - 4 = 27
// 0x1020 - (0x1005 + 1) - 4 = 22
// 0x1028 - (0x100a + 1) - 4 = 25
// 0x1030 - (0x100f + 1) - 4 = 28

// DISASM:      _start:
// DISASM-NEXT:   1000:  e9 {{.*}}       jmp  27
// DISASM-NEXT:   1005:  e9 {{.*}}       jmp  22
// DISASM-NEXT:   100a:  e9 {{.*}}       jmp  25
// DISASM-NEXT:   100f:  e9 {{.*}}       jmp  28

// 0x20A0 - 0x1026  = 4218
// 0x20A8 - 0x102e  = 4218
// 0x20B0 - 0x1036  = 4218

// DISASM:      Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   1020:  ff 25 {{.*}}       jmpq *4218(%rip)
// DISASM-NEXT:   1026:  90                 nop
// DISASM-NEXT:   1027:  90                 nop
// DISASM-NEXT:   1028:  ff 25 {{.*}}       jmpq *4218(%rip)
// DISASM-NEXT:   102e:  90                 nop
// DISASM-NEXT:   102f:  90                 nop
// DISASM-NEXT:   1030:  ff 25 {{.*}}       jmpq *4218(%rip)
// DISASM-NEXT:   1036:  90                 nop
// DISASM-NEXT:   1037:  90                 nop

// 0x11020 - (0x11000 + 1) - 4 = 27
// 0x11020 - (0x11005 + 1) - 4 = 22
// 0x11028 - (0x1100a + 1) - 4 = 25
// 0x11000 - (0x1100f + 1) - 4 = -20

// DISASM2:      _start:
// DISASM2-NEXT:   11000:  e9 {{.*}}     jmp  27
// DISASM2-NEXT:   11005:  e9 {{.*}}     jmp  22
// DISASM2-NEXT:   1100a:  e9 {{.*}}     jmp  25
// DISASM2-NEXT:   1100f:  e9 {{.*}}     jmp  -20

// DISASM2:      Disassembly of section .plt:
// DISASM2-NEXT: .plt:
// DISASM2-NEXT:     11020:  ff 25 7a 10 00 00 jmpq *4218(%rip)
// DISASM2-NEXT:     11026:  90                nop
// DISASM2-NEXT:     11027:  90                nop
// DISASM2-NEXT:     11028:  ff 25 7a 10 00 00 jmpq *4218(%rip)
// DISASM2-NEXT:     1102e:  90                nop
// DISASM2-NEXT:     1102f:  90                nop
// DISASM2-NEXT-NOT: 110C0

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
  jmp _start@plt
