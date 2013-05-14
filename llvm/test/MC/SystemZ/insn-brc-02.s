# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: offset out of range
#CHECK: brc	0, -0x100002
#CHECK: error: offset out of range
#CHECK: brc	0, -1
#CHECK: error: offset out of range
#CHECK: brc	0, 1
#CHECK: error: offset out of range
#CHECK: brc	0, 0x10000

	brc	0, -0x100002
	brc	0, -1
	brc	0, 1
	brc	0, 0x10000

#CHECK: error: invalid operand
#CHECK: brc	foo, bar
#CHECK: error: invalid operand
#CHECK: brc	-1, bar
#CHECK: error: invalid operand
#CHECK: brc	16, bar

	brc	foo, bar
	brc	-1, bar
	brc	16, bar
