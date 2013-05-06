# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: sl	%r0, -1
#CHECK: error: invalid operand
#CHECK: sl	%r0, 4096

	sl	%r0, -1
	sl	%r0, 4096
