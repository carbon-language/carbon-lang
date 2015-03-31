# For zEC12 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=zEC12 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,256,0,0

	risbgn	%r0,%r0,0,0,-1
	risbgn	%r0,%r0,0,0,64
	risbgn	%r0,%r0,0,-1,0
	risbgn	%r0,%r0,0,256,0
	risbgn	%r0,%r0,-1,0,0
	risbgn	%r0,%r0,256,0,0

