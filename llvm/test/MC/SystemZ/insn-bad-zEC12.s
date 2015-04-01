# For zEC12 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=zEC12 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: ntstg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ntstg	%r0, 524288

	ntstg	%r0, -524289
	ntstg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: ppa	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: ppa	%r0, %r0, 16

	ppa	%r0, %r0, -1
	ppa	%r0, %r0, 16

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

#CHECK: error: invalid operand
#CHECK: tabort	-1
#CHECK: error: invalid operand
#CHECK: tabort	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: tabort	0(%r1,%r2)

	tabort	-1
	tabort	4096
	tabort	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: tbegin	-1, 0
#CHECK: error: invalid operand
#CHECK: tbegin	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: tbegin	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: tbegin	0, -1
#CHECK: error: invalid operand
#CHECK: tbegin	0, 65536

	tbegin	-1, 0
	tbegin	4096, 0
	tbegin	0(%r1,%r2), 0
	tbegin	0, -1
	tbegin	0, 65536

#CHECK: error: invalid operand
#CHECK: tbeginc	-1, 0
#CHECK: error: invalid operand
#CHECK: tbeginc	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: tbeginc	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: tbeginc	0, -1
#CHECK: error: invalid operand
#CHECK: tbeginc	0, 65536

	tbeginc	-1, 0
	tbeginc	4096, 0
	tbeginc	0(%r1,%r2), 0
	tbeginc	0, -1
	tbeginc	0, 65536
