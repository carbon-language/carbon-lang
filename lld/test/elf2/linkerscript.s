# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "GROUP(" %t ")" > %t.script
# RUN: lld -flavor gnu2 -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(" %t.script2 ")" > %t.script1
# RUN: echo "GROUP(" %t ")" > %t.script2
# RUN: lld -flavor gnu2 -o %t2 %t.script1
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "OUTPUT_FORMAT(\"elf64-x86-64\") /*/*/ GROUP(" %t ")" > %t.script
# RUN: lld -flavor gnu2 -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "GROUP(AS_NEEDED(" %t "))" > %t.script
# RUN: lld -flavor gnu2 -o %t2 %t.script
# RUN: llvm-readobj %t2 > /dev/null

# RUN: echo "FOO(BAR)" > %t.script
# RUN: not lld -flavor gnu2 -o foo %t.script > %t.log 2>&1
# RUN: FileCheck -check-prefix=ERR1 %s < %t.log

# ERR1: unknown directive: FOO

.globl _start;
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
