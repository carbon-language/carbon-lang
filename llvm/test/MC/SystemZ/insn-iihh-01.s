# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: iihh	%r0, 0                  # encoding: [0xa5,0x00,0x00,0x00]
#CHECK: iihh	%r0, 32768              # encoding: [0xa5,0x00,0x80,0x00]
#CHECK: iihh	%r0, 65535              # encoding: [0xa5,0x00,0xff,0xff]
#CHECK: iihh	%r15, 0                 # encoding: [0xa5,0xf0,0x00,0x00]

	iihh	%r0, 0
	iihh	%r0, 0x8000
	iihh	%r0, 0xffff
	iihh	%r15, 0
