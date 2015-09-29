# REQUIRES: x86
# Verify that default output filename is a.out.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir
# RUN: not test -e a.out
# RUN: lld -flavor gnu2 %t
# RUN: test -e a.out

.globl _start;
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
