# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: cghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: cghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: cghi	%r0, foo

	cghi	%r0, -32769
	cghi	%r0, 32768
	cghi	%r0, foo
