# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t
# RUN: ld.lld2 %t -o %t2
# RUN: llvm-objdump -d %t2 | FileCheck %s
# REQUIRES: aarch64

.section .R_AARCH64_ADR_PREL_LO21,"ax",@progbits
.globl _start
_start:
  adr x1,msg
msg:  .asciz  "Hello, world\n"
msgend:

# CHECK: Disassembly of section .R_AARCH64_ADR_PREL_LO21:
# CHECK: _start:
# CHECK:        0:       21 00 00 10     adr     x1, #4
# CHECK: msg:
# CHECK:        4:
# #4 is the adr immediate value.

.section .R_AARCH64_ADR_PREL_PG_H121,"ax",@progbits
  adrp x1,mystr
mystr:
  .asciz "blah"
  .size mystr, 4

# S = 0x11012, A = 0x4, P = 0x11012
# PAGE(S + A) = 0x11000
# PAGE(P) = 0x11000
#
# CHECK: Disassembly of section .R_AARCH64_ADR_PREL_PG_H121:
# CHECK-NEXT: $x.2:
# CHECK-NEXT:   401012:       01 00 00 90     adrp    x1, #0

.section .R_AARCH64_ADD_ABS_LO12_NC,"ax",@progbits
  add x0, x0, :lo12:.L.str
.L.str:
  .asciz "blah"
  .size mystr, 4

# S = 0x1101b, A = 0x4
# R = (S + A) & 0xFFF = 0x1f
# R << 10 = 0x7c00
#
# CHECK: Disassembly of section .R_AARCH64_ADD_ABS_LO12_NC:
# CHECK-NEXT: $x.4:
# CHECK-NEXT:   40101b:       00 7c 00 91     add     x0, x0, #31
