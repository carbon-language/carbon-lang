# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: mseb	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mseb	%f0, %f0, 4096

	mseb	%f0, %f0, -1
	mseb	%f0, %f0, 4096
