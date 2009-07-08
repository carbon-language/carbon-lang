# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .byte 10
TEST0:  
        .fill 1, 1, 10

# CHECK: TEST1:
# CHECK: .short 3
# CHECK: .short 3
TEST1:  
        .fill 2, 2, 3
