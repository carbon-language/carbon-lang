# RUN: llvm-mc -triple i386-unknown-unknown %s 2> %t.err | FileCheck %s
# RUN: FileCheck --check-prefix=CHECK-WARNINGS %s < %t.err

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

# CHECK: TEST5
# CHECK: .short  2
# CHECK: .byte   0
# CHECK: .short  2
# CHECK: .byte   0
# CHECK: .short  2
# CHECK: .byte   0
# CHECK: .short  2
# CHECK: .byte   0
TEST5:
	.fill 4, 3, 2

# CHECK: TEST6
# CHECK: .long 2
# CHECK: .long 0
# CHECK-WARNINGS: '.fill' directive with size greater than 8 has been truncated to 8
TEST6:
	.fill 1, 9, 2

# CHECK: TEST7
# CHECK: .long 0
# CHECK: .long 0
# CHECK-WARNINGS: '.fill' directive pattern has been truncated to 32-bits
TEST7:
	.fill 1, 8, 1<<32

# CHECK-WARNINGS: '.fill' directive with negative repeat count has no effect
TEST8:
	.fill -1, 8, 1

# CHECK-WARNINGS: '.fill' directive with negative size has no effect
TEST9:
	.fill 1, -1, 1

# CHECK: TEST10
# CHECK: .short  22136
# CHECK: .byte   52
TEST10:
	.fill 1, 3, 0x12345678
