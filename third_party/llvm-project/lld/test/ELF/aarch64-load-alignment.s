# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-none %s -o %t.o
# RUN: not ld.lld -shared %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: improper alignment for relocation R_AARCH64_LD_PREL_LO19: 0x2007D is not aligned to 4 bytes

  ldr x8, patatino
  .data
  .zero 5
patatino:
