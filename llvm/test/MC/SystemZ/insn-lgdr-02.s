# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: lgdr	%f0, %f0
#CHECK: error: invalid register
#CHECK: lgdr	%r0, %r0
#CHECK: error: invalid register
#CHECK: lgdr	%r0, %a0
#CHECK: error: invalid register
#CHECK: lgdr	%a0, %f0

	lgdr	%f0, %f0
	lgdr	%r0, %r0
	lgdr	%r0, %a0
	lgdr	%a0, %f0
