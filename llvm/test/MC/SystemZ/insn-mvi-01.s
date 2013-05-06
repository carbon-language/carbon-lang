# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: mvi	0, 0                    # encoding: [0x92,0x00,0x00,0x00]
#CHECK: mvi	4095, 0                 # encoding: [0x92,0x00,0x0f,0xff]
#CHECK: mvi	0, 255                  # encoding: [0x92,0xff,0x00,0x00]
#CHECK: mvi	0(%r1), 42              # encoding: [0x92,0x2a,0x10,0x00]
#CHECK: mvi	0(%r15), 42             # encoding: [0x92,0x2a,0xf0,0x00]
#CHECK: mvi	4095(%r1), 42           # encoding: [0x92,0x2a,0x1f,0xff]
#CHECK: mvi	4095(%r15), 42          # encoding: [0x92,0x2a,0xff,0xff]

	mvi	0, 0
	mvi	4095, 0
	mvi	0, 255
	mvi	0(%r1), 42
	mvi	0(%r15), 42
	mvi	4095(%r1), 42
	mvi	4095(%r15), 42
