# Check R_MIPS_32 relocation calculation.
# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t.o
# RUN: ld.lld2 %t.o -o %t.exe
# RUN: llvm-objdump -s -t %t.exe | FileCheck %s

# REQUIRES: mips

  .globl  __start
__start:
  nop

  .data
  .type  v1,@object
  .size  v1,4
v1:
  .word 0

  .globl v2
  .type  v2,@object
  .size  v2,8
v2:
  .word v2+4 # R_MIPS_32 target v2 addend 4
  .word v1   # R_MIPS_32 target v1 addend 0

# CHECK: Contents of section .data:
# CHECK-NEXT: 30000 00000000 08000300 00000300
#                            ^-- v2+4 ^-- v1

# CHECK: SYMBOL TABLE:
# CHECK: 00030000 l       .data           00000004 v1
# CHECK: 00030004 g       .data           00000008 v2
