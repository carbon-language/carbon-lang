# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: dsg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dsg	%r0, 524288
#CHECK: error: invalid register
#CHECK: dsg	%r1, 0
#CHECK: error: invalid register
#CHECK: dsg	%r15, 0

	dsg	%r0, -524289
	dsg	%r0, 524288
	dsg	%r1, 0
	dsg	%r15, 0
