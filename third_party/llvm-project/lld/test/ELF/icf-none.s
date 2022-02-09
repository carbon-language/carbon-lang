# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o /dev/null --icf=all --icf=none --print-icf-sections | count 0

.globl _start, f1, f2
_start:
  ret

.section .text.f1, "ax"
f1:
  mov $60, %rax
  mov $42, %rdi
  syscall

.section .text.f2, "ax"
f2:
  mov $60, %rax
  mov $42, %rdi
  syscall
