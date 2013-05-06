# RUN: not llvm-mc -triple s390x-linux-gnu < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register
#CHECK: .cfi_offset %a0,0
#CHECK: error: register expected
#CHECK: .cfi_offset %foo,0
#CHECK: error: register expected
#CHECK: .cfi_offset %,0
#CHECK: error: register expected
#CHECK: .cfi_offset r0,0

	.cfi_startproc
	.cfi_offset %a0,0
	.cfi_offset %foo,0
	.cfi_offset %,0
	.cfi_offset r0,0
	.cfi_endproc
