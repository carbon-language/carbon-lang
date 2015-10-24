# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld2 %t.o -o %t1
#RUN: llvm-objdump -section-headers %t1 | FileCheck %s -check-prefix BEFORE
#BEFORE:       4 .symtab 00000030
#BEFORE-NEXT:  5 .shstrtab 0000002c
#BEFORE-NEXT:  6 .strtab 00000008

#RUN: ld.lld2 %t.o --strip-all -o %t1
#RUN: llvm-objdump -section-headers %t1 | FileCheck %s -check-prefix AFTER
#AFTER-NOT: .symtab
#AFTER: 4 .shstrtab 0000001c
#AFTER-NOT: .strtab

# Test alias -s
#RUN: ld.lld2 %t.o -s -o %t1
#RUN: llvm-objdump -section-headers %t1 | FileCheck %s -check-prefix AFTER

# exits with return code 42 on linux
.globl _start;
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
