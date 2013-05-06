# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ear	%r0, %a0                # encoding: [0xb2,0x4f,0x00,0x00]
#CHECK: ear	%r0, %a15               # encoding: [0xb2,0x4f,0x00,0x0f]
#CHECK: ear	%r15, %a0               # encoding: [0xb2,0x4f,0x00,0xf0]
#CHECK: ear	%r7, %a8                # encoding: [0xb2,0x4f,0x00,0x78]
#CHECK: ear	%r15, %a15              # encoding: [0xb2,0x4f,0x00,0xff]

	ear	%r0, %a0
	ear	%r0, %a15
	ear	%r15, %a0
	ear	%r7, %a8
	ear	%r15, %a15
