@ RUN: not llvm-mc -triple armv7a--none-eabi -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

  .text
@ CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: symbol 'undef' can not be undefined in a subtraction expression
  .word (0-undef)
