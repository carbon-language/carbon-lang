# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: alfi	%r0, -1
#CHECK: error: invalid operand
#CHECK: alfi	%r0, (1 << 32)

	alfi	%r0, -1
	alfi	%r0, (1 << 32)
