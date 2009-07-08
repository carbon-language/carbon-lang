# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
TEST0:  
        .ascii

# CHECK: TEST1:
TEST1:  
        .asciz

# CHECK: TEST2:
# CHECK: .byte 65
TEST2:  
        .ascii "A"

# CHECK: TEST3:
# CHECK: .byte 66
# CHECK: .byte 0
# CHECK: .byte 67
# CHECK: .byte 0
TEST3:  
        .asciz "B", "C"

       