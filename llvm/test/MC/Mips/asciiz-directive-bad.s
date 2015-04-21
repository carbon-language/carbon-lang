# RUN: not llvm-mc -triple mips-unknown-linux %s 2>&1 | FileCheck %s

  .asciiz 12
# CHECK: :[[@LINE-1]]:11: error: expected string in '.asciiz' directive
  .asciiz "a"3
# CHECK: :[[@LINE-1]]:14: error: unexpected token in '.asciiz' directive
  .asciiz "a",
# CHECK: :[[@LINE-1]]:15: error: expected string in '.asciiz' directive
