# RUN: llvm-mc -triple x86_64-pc-linux-gnu %s -filetype=obj -o %t.o
# RUN: llvm-readobj -s --elf-output-style=GNU %t.o | FileCheck %s

## Check we are able to set the custom flag ('E') for debug sections.
# CHECK: .debug_info   {{.*}}  E
# CHECK: .debug_str    {{.*}}  EMS

.section .debug_info,"e"
nop

.section .debug_str,"eMS",@progbits,1
nop
