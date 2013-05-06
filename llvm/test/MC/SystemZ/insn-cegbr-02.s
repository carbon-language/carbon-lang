# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: cegbr	%r0, %r0
#CHECK: error: invalid register
#CHECK: cegbr	%f0, %f0
#CHECK: error: invalid register
#CHECK: cegbr	%f0, %a0
#CHECK: error: invalid register
#CHECK: cegbr	%a0, %r0

	cegbr	%r0, %r0
	cegbr	%f0, %f0
	cegbr	%f0, %a0
	cegbr	%a0, %r0
