# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ic	%r0, 0                  # encoding: [0x43,0x00,0x00,0x00]
#CHECK: ic	%r0, 4095               # encoding: [0x43,0x00,0x0f,0xff]
#CHECK: ic	%r0, 0(%r1)             # encoding: [0x43,0x00,0x10,0x00]
#CHECK: ic	%r0, 0(%r15)            # encoding: [0x43,0x00,0xf0,0x00]
#CHECK: ic	%r0, 4095(%r1,%r15)     # encoding: [0x43,0x01,0xff,0xff]
#CHECK: ic	%r0, 4095(%r15,%r1)     # encoding: [0x43,0x0f,0x1f,0xff]
#CHECK: ic	%r15, 0                 # encoding: [0x43,0xf0,0x00,0x00]

	ic	%r0, 0
	ic	%r0, 4095
	ic	%r0, 0(%r1)
	ic	%r0, 0(%r15)
	ic	%r0, 4095(%r1,%r15)
	ic	%r0, 4095(%r15,%r1)
	ic	%r15, 0
