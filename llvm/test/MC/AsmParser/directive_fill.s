# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .byte 10
TEST0:  
        .fill 1, 1, 10

# CHECK: TEST1:
# CHECK: .short 3
# CHECK: .short 3
TEST1:  
        .fill 2, 2, 3

# CHECK: TEST2:
# CHECK: .quad 4
TEST2:  
        .fill 1, 8, 4

# CHECK: TEST3
# CHECK: .byte 0
# CHECK: .byte 0
# CHECK: .byte 0
# CHECK: .byte 0
TEST3:
	.fill 4

# CHECK: TEST4
# CHECK: .short 0
# CHECK: .short 0
# CHECK: .short 0
# CHECK: .short 0
TEST4:
	.fill 4, 2
