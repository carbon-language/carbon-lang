# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: iilh	%r0, 0                  # encoding: [0xa5,0x02,0x00,0x00]
#CHECK: iilh	%r0, 32768              # encoding: [0xa5,0x02,0x80,0x00]
#CHECK: iilh	%r0, 65535              # encoding: [0xa5,0x02,0xff,0xff]
#CHECK: iilh	%r15, 0                 # encoding: [0xa5,0xf2,0x00,0x00]

	iilh	%r0, 0
	iilh	%r0, 0x8000
	iilh	%r0, 0xffff
	iilh	%r15, 0
