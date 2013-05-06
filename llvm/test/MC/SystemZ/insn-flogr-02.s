# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: flogr	%r1, %r0
#CHECK: error: invalid register
#CHECK: flogr	%r15, %r0

	flogr	%r1, %r0
	flogr	%r15, %r0
