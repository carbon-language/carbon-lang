// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/shared.s -o %t2
// RUN: ld.lld %t2 -soname fixed-length-string.so -o %t2.so -shared
// RUN: ld.lld --hash-style=sysv %t %t2.so -o %t3
// RUN: llvm-readobj -S  %t3 | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump -s -d %t3 | FileCheck %s

// SEC:      Name: .plt
// SEC-NEXT: Type: SHT_PROGBITS
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_EXECINSTR
// SEC-NEXT: ]
// SEC-NEXT: Address: 0x201340
// SEC-NEXT: Offset: 0x340
// SEC-NEXT: Size: 48

// SEC:         Name: .got
// SEC-NEXT:   Type: SHT_PROGBITS
// SEC-NEXT:   Flags [
// SEC-NEXT:     SHF_ALLOC
// SEC-NEXT:     SHF_WRITE
// SEC-NEXT:   ]
// SEC-NEXT:   Address: 0x202460
// SEC-NEXT:   Offset:
// SEC-NEXT:   Size: 8
// SEC-NEXT:   Link: 0
// SEC-NEXT:   Info: 0
// SEC-NEXT:   AddressAlignment: 8
// SEC-NEXT:   EntrySize: 0
// SEC-NEXT: }

// SEC:        Name: .got.plt
// SEC-NEXT:   Type: SHT_PROGBITS
// SEC-NEXT:   Flags [
// SEC-NEXT:     SHF_ALLOC
// SEC-NEXT:     SHF_WRITE
// SEC-NEXT:   ]
// SEC-NEXT:   Address: 0x203468
// SEC-NEXT:   Offset: 0x468
// SEC-NEXT:   Size: 40
// SEC-NEXT:   Link: 0
// SEC-NEXT:   Info: 0
// SEC-NEXT:   AddressAlignment: 8
// SEC-NEXT:   EntrySize: 0
// SEC-NEXT:   }

.section .R_X86_64_64,"a",@progbits
.global R_X86_64_64
R_X86_64_64:
 .quad R_X86_64_64

// CHECK:      Contents of section .R_X86_64_64:
// CHECK-NEXT:   2002f8 f8022000 00000000

.section .R_X86_64_GOTPCREL,"a",@progbits
.global R_X86_64_GOTPCREL
R_X86_64_GOTPCREL:
 .long zed@gotpcrel

// 0x202460(.got) - 0x200300(.R_X86_64_GOTPCREL) = 0x2160
// CHECK:      Contents of section .R_X86_64_GOTPCREL
// CHECK-NEXT:   200300 60210000

.section .R_X86_64_GOT32,"a",@progbits
.global R_X86_64_GOT32
R_X86_64_GOT32:
        .long zed@got

// CHECK: Contents of section .R_X86_64_GOT32:
// CHECK-NEXT: f8efffff


// CHECK: Contents of section .R_X86_64_GOT64:
// CHECK-NEXT: f8efffff ffffffff
.section .R_X86_64_GOT64,"a",@progbits
.global R_X86_64_GOT64
R_X86_64_GOT64:
        .quad zed@got

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
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:   201310:  e8 04 00 00 00   callq 0x201319
// CHECK-NEXT:   201315:

// CHECK:      <lulz>:
// CHECK-NEXT:   201319:  90  nop


.section       .text2,"ax",@progbits
.global R_X86_64_32
R_X86_64_32:
  movl $R_X86_64_32, %edx

// FIXME: this would be far more self evident if llvm-objdump printed
// constants in hex.
// CHECK: Disassembly of section .text2:
// CHECK-EMPTY:
// CHECK-NEXT: <R_X86_64_32>:
// CHECK-NEXT:  20131a: {{.*}} movl $2102042, %edx

.section .R_X86_64_32S,"ax",@progbits
.global R_X86_64_32S
R_X86_64_32S:
  movq lulz - 0x100000, %rdx

// CHECK: Disassembly of section .R_X86_64_32S:
// CHECK-EMPTY:
// CHECK-NEXT: <R_X86_64_32S>:
// CHECK-NEXT:  {{.*}}: {{.*}} movq 1053465, %rdx

.section .R_X86_64_PC32,"ax",@progbits
.global R_X86_64_PC32
R_X86_64_PC32:
 call bar
 movl $bar, %eax
//16 is a size of PLT[0]
// 0x201340 + 16 - (0x201327 + 5) = 36
// CHECK:      Disassembly of section .R_X86_64_PC32:
// CHECK-EMPTY:
// CHECK-NEXT: <R_X86_64_PC32>:
// CHECK-NEXT:  201327:   {{.*}}  callq  0x201350
// CHECK-NEXT:  20132c:   {{.*}}  movl $2102096, %eax

.section .R_X86_64_32S_2,"ax",@progbits
.global R_X86_64_32S_2
R_X86_64_32S_2:
  mov bar2, %eax
// plt is  at 0x201340. The second plt entry is at 0x201360 == 2102112
// CHECK:      Disassembly of section .R_X86_64_32S_2:
// CHECK-EMPTY:
// CHECK-NEXT: <R_X86_64_32S_2>:
// CHECK-NEXT: 201331: {{.*}}  movl    2102112, %eax
