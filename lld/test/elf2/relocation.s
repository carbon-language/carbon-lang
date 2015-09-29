// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/shared.s -o %t2
// RUN: lld -flavor gnu2 %t2 -o %t2.so -shared
// RUN: lld -flavor gnu2 %t %t2.so -o %t3
// RUN: llvm-readobj -s  %t3 | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump -s -d %t3 | FileCheck %s
// REQUIRES: x86

// SEC:         Name: .got
// SEC-NEXT:   Type: SHT_PROGBITS
// SEC-NEXT:   Flags [
// SEC-NEXT:     SHF_ALLOC
// SEC-NEXT:     SHF_WRITE
// SEC-NEXT:   ]
// SEC-NEXT:   Address: 0x15000
// SEC-NEXT:   Offset:
// SEC-NEXT:   Size: 16
// SEC-NEXT:   Link: 0
// SEC-NEXT:   Info: 0
// SEC-NEXT:   AddressAlignment: 8
// SEC-NEXT:   EntrySize: 0
// SEC-NEXT: }

// SEC:      Name: .plt
// SEC-NEXT: Type: SHT_PROGBITS
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_EXECINSTR
// SEC-NEXT: ]
// SEC-NEXT: Address: 0x16000
// SEC-NEXT: Offset: 0x6000
// SEC-NEXT: Size: 8

.section       .text,"ax",@progbits,unique,1
.global _start
_start:
  call lulz

.section       .text,"ax",@progbits,unique,2
.zero 4
.global lulz
lulz:
  nop

// CHECK: Disassembly of section .text:
// CHECK-NEXT: _start:
// CHECK-NEXT:   11000:  e8 04 00 00 00   callq 4
// CHECK-NEXT:   11005:

// CHECK:      lulz:
// CHECK-NEXT:   11009:  90  nop


.section       .text2,"ax",@progbits
.global R_X86_64_32
R_X86_64_32:
  movl $R_X86_64_32, %edx

// FIXME: this would be far more self evident if llvm-objdump printed
// constants in hex.
// CHECK: Disassembly of section .text2:
// CHECK-NEXT: R_X86_64_32:
// CHECK-NEXT:  1100c: {{.*}} movl $69644, %edx

.section .R_X86_64_32S,"ax",@progbits
.global R_X86_64_32S
R_X86_64_32S:
  movq lulz - 0x100000, %rdx

// CHECK: Disassembly of section .R_X86_64_32S:
// CHECK-NEXT: R_X86_64_32S:
// CHECK-NEXT:  {{.*}}: {{.*}} movq -978935, %rdx

.section .R_X86_64_PC32,"ax",@progbits
.global R_X86_64_PC32
R_X86_64_PC32:
 call bar
// 0x16000 - (0x11019 + 5) = 20450
// CHECK:      Disassembly of section .R_X86_64_PC32:
// CHECK-NEXT: R_X86_64_PC32:
// CHECK-NEXT:  11019:   e8 e2 4f 00 00  callq  20450

.section .R_X86_64_64,"a",@progbits
.global R_X86_64_64
R_X86_64_64:
 .quad R_X86_64_64

// CHECK:      Contents of section .R_X86_64_64:
// CHECK-NEXT:   12000 00200100 00000000

.section .R_X86_64_GOTPCREL,"a",@progbits
.global R_X86_64_GOTPCREL
R_X86_64_GOTPCREL:
 .long R_X86_64_GOTPCREL@gotpcrel

// 0x15008 - 0x12008 = 12288
// 12288 = 0x00300000   in little endian
// CHECK:      Contents of section .R_X86_64_GOTPCREL
// CHECK-NEXT:   12008 00300000
