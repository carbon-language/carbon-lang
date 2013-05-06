# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: stc	%r0, -1
#CHECK: error: invalid operand
#CHECK: stc	%r0, 4096

	stc	%r0, -1
	stc	%r0, 4096
