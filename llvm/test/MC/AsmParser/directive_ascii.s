# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .data
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
        
# CHECK: TEST4:
# CHECK: .byte 1
# CHECK: .byte 1
# CHECK: .byte 7
# CHECK: .byte 0
# CHECK: .byte 56
# CHECK: .byte 1
# CHECK: .byte 0
# CHECK: .byte 49
# CHECK: .byte 0
TEST4:  
        .ascii "\1\01\07\08\001\0001\b\0"
        
# CHECK: TEST5:
# CHECK: .byte 8
# CHECK: .byte 12
# CHECK: .byte 10
# CHECK: .byte 13
# CHECK: .byte 9
# CHECK: .byte 92
# CHECK: .byte 34
TEST5:
        .ascii "\b\f\n\r\t\\\""
        
