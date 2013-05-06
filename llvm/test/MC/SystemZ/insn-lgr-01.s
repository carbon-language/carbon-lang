# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgr	%r0, %r9                # encoding: [0xb9,0x04,0x00,0x09]
#CHECK: lgr	%r0, %r15               # encoding: [0xb9,0x04,0x00,0x0f]
#CHECK: lgr	%r15, %r0               # encoding: [0xb9,0x04,0x00,0xf0]
#CHECK: lgr	%r15, %r9               # encoding: [0xb9,0x04,0x00,0xf9]

	lgr	%r0,%r9
	lgr	%r0,%r15
	lgr	%r15,%r0
	lgr	%r15,%r9
