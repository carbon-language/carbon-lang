# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lr	%r0, %r9                # encoding: [0x18,0x09]
#CHECK: lr	%r0, %r15               # encoding: [0x18,0x0f]
#CHECK: lr	%r15, %r0               # encoding: [0x18,0xf0]
#CHECK: lr	%r15, %r9               # encoding: [0x18,0xf9]

	lr	%r0,%r9
	lr	%r0,%r15
	lr	%r15,%r0
	lr	%r15,%r9
