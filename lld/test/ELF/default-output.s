# REQUIRES: x86
# Verify that default output filename is a.out.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: mkdir -p %t.dir
# RUN: cd %t.dir
# RUN: rm -f a.out
# RUN: not llvm-readobj a.out > /dev/null 2>&1
# RUN: ld.lld %t
# RUN: llvm-readobj a.out > /dev/null 2>&1

# RUN: not ld.lld %t -o 2>&1 | FileCheck --check-prefix=NO_O_VAL %s
# NO_O_VAL: error: -o: missing argument

.globl _start
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
