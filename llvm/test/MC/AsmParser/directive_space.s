# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .byte 0
TEST0:  
        .space 1

# CHECK: TEST1:
# CHECK: .byte 3
# CHECK: .byte 3
TEST1:  
        .space 2, 3
