# RUN: llvm-mc -filetype=obj -triple=i386-unknown-elf %s | llvm-objdump -t - | FileCheck %s

# CHECK: 00000000         .text           00000000 TEST0
TEST0:
  .fill 0x10
# CHECK: 00000010         .text           00000000 TEST1
TEST1:
  .fill TEST1 - TEST0 + 0x5
# CHECK: 00000025         .text           00000000 TEST2
TEST2:
  .zero TEST2 - (TEST1 + 0x5)
# CHECK: 00000035         .text           00000000 TEST3
TEST3:
  .skip (TEST1 - TEST0) * 2
# CHECK: 00000055         .text           00000000 TEST4
TEST4:
  .space TEST2 - TEST1, 1
