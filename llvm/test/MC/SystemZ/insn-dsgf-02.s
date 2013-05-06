# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: dsgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dsgf	%r0, 524288
#CHECK: error: invalid register
#CHECK: dsgf	%r1, 0
#CHECK: error: invalid register
#CHECK: dsgf	%r15, 0

	dsgf	%r0, -524289
	dsgf	%r0, 524288
	dsgf	%r1, 0
	dsgf	%r15, 0
