# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld2 -shared %t -o %t.so
# RUN: ld.lld2 -z now -Bsymbolic %t %t.so -o %t1
# RUN: ld.lld2 %t %t.so -o %t2
# RUN: llvm-readobj -dynamic-table %t1 | FileCheck -check-prefix=FLAGS %s
# RUN: llvm-readobj -dynamic-table %t2 | FileCheck %s

# FLAGS: DynamicSection [
# FLAGS:   0x000000006FFFFFFB FLAGS_1 NOW GLOBAL
# FLAGS: ]

# CHECK: DynamicSection [
# CHECK-NOT:   0x000000006FFFFFFB FLAGS_1 NOW GLOBAL
# CHECK: ]

.globl _start
_start:
