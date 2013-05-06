# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: la	%r0, 0                  # encoding: [0x41,0x00,0x00,0x00]
#CHECK: la	%r0, 4095               # encoding: [0x41,0x00,0x0f,0xff]
#CHECK: la	%r0, 0(%r1)             # encoding: [0x41,0x00,0x10,0x00]
#CHECK: la	%r0, 0(%r15)            # encoding: [0x41,0x00,0xf0,0x00]
#CHECK: la	%r0, 4095(%r1,%r15)     # encoding: [0x41,0x01,0xff,0xff]
#CHECK: la	%r0, 4095(%r15,%r1)     # encoding: [0x41,0x0f,0x1f,0xff]
#CHECK: la	%r15, 0                 # encoding: [0x41,0xf0,0x00,0x00]

	la	%r0, 0
	la	%r0, 4095
	la	%r0, 0(%r1)
	la	%r0, 0(%r15)
	la	%r0, 4095(%r1,%r15)
	la	%r0, 4095(%r15,%r1)
	la	%r15, 0
