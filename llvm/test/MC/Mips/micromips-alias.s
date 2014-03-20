# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o - \
# RUN:   | llvm-readobj -t | FileCheck %s

# Symbol bar must be marked as micromips.
# CHECK: Name: bar
# CHECK: Other: 128
  .align 2
  .type  f,@function
  .set   nomips16
  .set   micromips
f:
  nop
  .set   nomicromips
  nop
  .globl bar
bar = f
