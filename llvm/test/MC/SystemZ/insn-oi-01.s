# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: oi	0, 0                    # encoding: [0x96,0x00,0x00,0x00]
#CHECK: oi	4095, 0                 # encoding: [0x96,0x00,0x0f,0xff]
#CHECK: oi	0, 255                  # encoding: [0x96,0xff,0x00,0x00]
#CHECK: oi	0(%r1), 42              # encoding: [0x96,0x2a,0x10,0x00]
#CHECK: oi	0(%r15), 42             # encoding: [0x96,0x2a,0xf0,0x00]
#CHECK: oi	4095(%r1), 42           # encoding: [0x96,0x2a,0x1f,0xff]
#CHECK: oi	4095(%r15), 42          # encoding: [0x96,0x2a,0xff,0xff]

	oi	0, 0
	oi	4095, 0
	oi	0, 255
	oi	0(%r1), 42
	oi	0(%r15), 42
	oi	4095(%r1), 42
	oi	4095(%r15), 42
