# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sl	%r0, 0                  # encoding: [0x5f,0x00,0x00,0x00]
#CHECK: sl	%r0, 4095               # encoding: [0x5f,0x00,0x0f,0xff]
#CHECK: sl	%r0, 0(%r1)             # encoding: [0x5f,0x00,0x10,0x00]
#CHECK: sl	%r0, 0(%r15)            # encoding: [0x5f,0x00,0xf0,0x00]
#CHECK: sl	%r0, 4095(%r1,%r15)     # encoding: [0x5f,0x01,0xff,0xff]
#CHECK: sl	%r0, 4095(%r15,%r1)     # encoding: [0x5f,0x0f,0x1f,0xff]
#CHECK: sl	%r15, 0                 # encoding: [0x5f,0xf0,0x00,0x00]

	sl	%r0, 0
	sl	%r0, 4095
	sl	%r0, 0(%r1)
	sl	%r0, 0(%r15)
	sl	%r0, 4095(%r1,%r15)
	sl	%r0, 4095(%r15,%r1)
	sl	%r15, 0
