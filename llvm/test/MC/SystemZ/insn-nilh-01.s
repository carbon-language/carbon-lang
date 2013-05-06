# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: nilh	%r0, 0                  # encoding: [0xa5,0x06,0x00,0x00]
#CHECK: nilh	%r0, 32768              # encoding: [0xa5,0x06,0x80,0x00]
#CHECK: nilh	%r0, 65535              # encoding: [0xa5,0x06,0xff,0xff]
#CHECK: nilh	%r15, 0                 # encoding: [0xa5,0xf6,0x00,0x00]

	nilh	%r0, 0
	nilh	%r0, 0x8000
	nilh	%r0, 0xffff
	nilh	%r15, 0
