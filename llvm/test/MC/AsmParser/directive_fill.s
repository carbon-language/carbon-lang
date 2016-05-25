# RUN: llvm-mc -triple i386-unknown-unknown %s 2> %t.err | FileCheck %s
# RUN: FileCheck --check-prefix=CHECK-WARNINGS %s < %t.err

# CHECK: TEST0:
# CHECK: .fill 1, 1, 0xa
TEST0:  
        .fill 1, 1, 10

# CHECK: TEST1:
# CHECK: .fill 2, 2, 0x3
TEST1:  
        .fill 2, 2, 3

# CHECK: TEST2:
# CHECK: .fill 1, 8, 0x4
TEST2:  
        .fill 1, 8, 4

# CHECK: TEST3
# CHECK: .fill 4
TEST3:
	.fill 4

# CHECK: TEST4
# CHECK: .fill 4, 2
TEST4:
	.fill 4, 2

# CHECK: TEST5
# CHECK: .fill 4, 3, 0x2
TEST5:
	.fill 4, 3, 2

# CHECK: TEST6
# CHECK: .fill 1, 8, 0x2
# CHECK-WARNINGS: '.fill' directive with size greater than 8 has been truncated to 8
TEST6:
	.fill 1, 9, 2

# CHECK: TEST7
# CHECK: .fill 1, 8, 0x100000000
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
# CHECK: .fill 1, 3, 0x12345678
TEST10:
	.fill 1, 3, 0x12345678

# CHECK: .fill TEST11-TEST10, 1, 0x0
TEST11:
  .fill TEST11 - TEST10

# CHECK: .fill TEST11-TEST12, 3, 0x12345678
TEST12:
  .fill TEST11 - TEST12, 3, 0x12345678
