# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: nihh	%r0, 0                  # encoding: [0xa5,0x04,0x00,0x00]
#CHECK: nihh	%r0, 32768              # encoding: [0xa5,0x04,0x80,0x00]
#CHECK: nihh	%r0, 65535              # encoding: [0xa5,0x04,0xff,0xff]
#CHECK: nihh	%r15, 0                 # encoding: [0xa5,0xf4,0x00,0x00]

	nihh	%r0, 0
	nihh	%r0, 0x8000
	nihh	%r0, 0xffff
	nihh	%r15, 0
