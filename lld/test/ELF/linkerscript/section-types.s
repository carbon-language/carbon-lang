# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { " > %t.script
# RUN: echo ".bar : { *(.foo) *(.init_array) }" >> %t.script
# RUN: echo "}" >> %t.script

# RUN: ld.lld -o %t1 --script %t.script %t

.section .foo,"aw"
  .quad 1

.section .init_array,"aw",@init_array
  .quad 0
