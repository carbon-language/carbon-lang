# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: offset out of range
#CHECK: brcl	0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: brcl	0, -1
#CHECK: error: offset out of range
#CHECK: brcl	0, 1
#CHECK: error: offset out of range
#CHECK: brcl	0, 0x100000000

	brcl	0, -0x1000000002
	brcl	0, -1
	brcl	0, 1
	brcl	0, 0x100000000

#CHECK: error: invalid operand
#CHECK: brcl	foo, bar
#CHECK: error: invalid operand
#CHECK: brcl	-1, bar
#CHECK: error: invalid operand
#CHECK: brcl	16, bar

	brcl	foo, bar
	brcl	-1, bar
	brcl	16, bar
