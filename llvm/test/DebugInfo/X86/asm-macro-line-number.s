# RUN: llvm-mc -g -triple i686-linux-gnu -filetype asm -o - %s | FileCheck %s

# 1 "reduced.S"
# 1 "<built-in>" 1
# 1 "reduced.S" 2

 .macro return arg
  movl %eax, \arg
  retl
 .endm

function:
 return 0

# CHECK: .file 2 "reduced.S"
# CHECK: .loc 2 8 0
# CHECK: movl %eax, 0
# CHECK: .loc 2 8 0
# CHECK: retl

