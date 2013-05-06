# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: oilh	%r0, 0                  # encoding: [0xa5,0x0a,0x00,0x00]
#CHECK: oilh	%r0, 32768              # encoding: [0xa5,0x0a,0x80,0x00]
#CHECK: oilh	%r0, 65535              # encoding: [0xa5,0x0a,0xff,0xff]
#CHECK: oilh	%r15, 0                 # encoding: [0xa5,0xfa,0x00,0x00]

	oilh	%r0, 0
	oilh	%r0, 0x8000
	oilh	%r0, 0xffff
	oilh	%r15, 0
