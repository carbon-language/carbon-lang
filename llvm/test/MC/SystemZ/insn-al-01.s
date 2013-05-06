# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: al	%r0, 0                  # encoding: [0x5e,0x00,0x00,0x00]
#CHECK: al	%r0, 4095               # encoding: [0x5e,0x00,0x0f,0xff]
#CHECK: al	%r0, 0(%r1)             # encoding: [0x5e,0x00,0x10,0x00]
#CHECK: al	%r0, 0(%r15)            # encoding: [0x5e,0x00,0xf0,0x00]
#CHECK: al	%r0, 4095(%r1,%r15)     # encoding: [0x5e,0x01,0xff,0xff]
#CHECK: al	%r0, 4095(%r15,%r1)     # encoding: [0x5e,0x0f,0x1f,0xff]
#CHECK: al	%r15, 0                 # encoding: [0x5e,0xf0,0x00,0x00]

	al	%r0, 0
	al	%r0, 4095
	al	%r0, 0(%r1)
	al	%r0, 0(%r15)
	al	%r0, 4095(%r1,%r15)
	al	%r0, 4095(%r15,%r1)
	al	%r15, 0
