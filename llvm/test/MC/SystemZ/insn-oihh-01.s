# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: oihh	%r0, 0                  # encoding: [0xa5,0x08,0x00,0x00]
#CHECK: oihh	%r0, 32768              # encoding: [0xa5,0x08,0x80,0x00]
#CHECK: oihh	%r0, 65535              # encoding: [0xa5,0x08,0xff,0xff]
#CHECK: oihh	%r15, 0                 # encoding: [0xa5,0xf8,0x00,0x00]

	oihh	%r0, 0
	oihh	%r0, 0x8000
	oihh	%r0, 0xffff
	oihh	%r15, 0
