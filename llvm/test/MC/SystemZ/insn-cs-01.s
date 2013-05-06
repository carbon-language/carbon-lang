# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cs	%r0, %r0, 0             # encoding: [0xba,0x00,0x00,0x00]
#CHECK: cs	%r0, %r0, 4095          # encoding: [0xba,0x00,0x0f,0xff]
#CHECK: cs	%r0, %r0, 0(%r1)        # encoding: [0xba,0x00,0x10,0x00]
#CHECK: cs	%r0, %r0, 0(%r15)       # encoding: [0xba,0x00,0xf0,0x00]
#CHECK: cs	%r0, %r0, 4095(%r1)     # encoding: [0xba,0x00,0x1f,0xff]
#CHECK: cs	%r0, %r0, 4095(%r15)    # encoding: [0xba,0x00,0xff,0xff]
#CHECK: cs	%r0, %r15, 0            # encoding: [0xba,0x0f,0x00,0x00]
#CHECK: cs	%r15, %r0, 0            # encoding: [0xba,0xf0,0x00,0x00]

	cs	%r0, %r0, 0
	cs	%r0, %r0, 4095
	cs	%r0, %r0, 0(%r1)
	cs	%r0, %r0, 0(%r15)
	cs	%r0, %r0, 4095(%r1)
	cs	%r0, %r0, 4095(%r15)
	cs	%r0, %r15, 0
	cs	%r15, %r0, 0
