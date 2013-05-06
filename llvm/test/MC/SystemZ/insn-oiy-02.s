# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: oiy	-524289, 0
#CHECK: error: invalid operand
#CHECK: oiy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: oiy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: oiy	0, -1
#CHECK: error: invalid operand
#CHECK: oiy	0, 256

	oiy	-524289, 0
	oiy	524288, 0
	oiy	0(%r1,%r2), 0
	oiy	0, -1
	oiy	0, 256
