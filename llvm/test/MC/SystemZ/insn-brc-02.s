# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: brc	foo, bar
#CHECK: error: invalid operand
#CHECK: brc	-1, bar
#CHECK: error: invalid operand
#CHECK: brc	16, bar

	brc	foo, bar
	brc	-1, bar
	brc	16, bar
