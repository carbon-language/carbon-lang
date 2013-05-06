# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: dsgfr	%r1, %r0
#CHECK: error: invalid register
#CHECK: dsgfr	%r15, %r0

	dsgfr	%r1, %r0
	dsgfr	%r15, %r0
