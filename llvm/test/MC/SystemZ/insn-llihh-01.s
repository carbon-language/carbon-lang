# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llihh	%r0, 0                  # encoding: [0xa5,0x0c,0x00,0x00]
#CHECK: llihh	%r0, 32768              # encoding: [0xa5,0x0c,0x80,0x00]
#CHECK: llihh	%r0, 65535              # encoding: [0xa5,0x0c,0xff,0xff]
#CHECK: llihh	%r15, 0                 # encoding: [0xa5,0xfc,0x00,0x00]

	llihh	%r0, 0
	llihh	%r0, 0x8000
	llihh	%r0, 0xffff
	llihh	%r15, 0
