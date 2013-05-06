# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llihl	%r0, 0                  # encoding: [0xa5,0x0d,0x00,0x00]
#CHECK: llihl	%r0, 32768              # encoding: [0xa5,0x0d,0x80,0x00]
#CHECK: llihl	%r0, 65535              # encoding: [0xa5,0x0d,0xff,0xff]
#CHECK: llihl	%r15, 0                 # encoding: [0xa5,0xfd,0x00,0x00]

	llihl	%r0, 0
	llihl	%r0, 0x8000
	llihl	%r0, 0xffff
	llihl	%r15, 0
