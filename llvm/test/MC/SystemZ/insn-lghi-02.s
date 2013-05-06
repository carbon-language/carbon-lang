# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: lghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: lghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: lghi	%r0, foo

	lghi	%r0, -32769
	lghi	%r0, 32768
	lghi	%r0, foo
