# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cli	0, 0                    # encoding: [0x95,0x00,0x00,0x00]
#CHECK: cli	4095, 0                 # encoding: [0x95,0x00,0x0f,0xff]
#CHECK: cli	0, 255                  # encoding: [0x95,0xff,0x00,0x00]
#CHECK: cli	0(%r1), 42              # encoding: [0x95,0x2a,0x10,0x00]
#CHECK: cli	0(%r15), 42             # encoding: [0x95,0x2a,0xf0,0x00]
#CHECK: cli	4095(%r1), 42           # encoding: [0x95,0x2a,0x1f,0xff]
#CHECK: cli	4095(%r15), 42          # encoding: [0x95,0x2a,0xff,0xff]

	cli	0, 0
	cli	4095, 0
	cli	0, 255
	cli	0(%r1), 42
	cli	0(%r15), 42
	cli	4095(%r1), 42
	cli	4095(%r15), 42
