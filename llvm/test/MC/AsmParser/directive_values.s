# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .byte 0
TEST0:  
        .byte 0

# CHECK: TEST1:
# CHECK: .short 3
TEST1:  
        .short 3

# CHECK: TEST2:
# CHECK: .long 8
TEST2:  
        .long 8

# CHECK: TEST3:
# CHECK: .quad 9
TEST3:  
        .quad 9

# CHECK: TEST4:
# CHECK: .short 3
TEST4:  
        .word 3
