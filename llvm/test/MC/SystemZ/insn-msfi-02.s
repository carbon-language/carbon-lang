# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: msfi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: msfi	%r0, (1 << 31)

	msfi	%r0, (-1 << 31) - 1
	msfi	%r0, (1 << 31)
