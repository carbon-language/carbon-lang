# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: xi	0, 0                    # encoding: [0x97,0x00,0x00,0x00]
#CHECK: xi	4095, 0                 # encoding: [0x97,0x00,0x0f,0xff]
#CHECK: xi	0, 255                  # encoding: [0x97,0xff,0x00,0x00]
#CHECK: xi	0(%r1), 42              # encoding: [0x97,0x2a,0x10,0x00]
#CHECK: xi	0(%r15), 42             # encoding: [0x97,0x2a,0xf0,0x00]
#CHECK: xi	4095(%r1), 42           # encoding: [0x97,0x2a,0x1f,0xff]
#CHECK: xi	4095(%r15), 42          # encoding: [0x97,0x2a,0xff,0xff]

	xi	0, 0
	xi	4095, 0
	xi	0, 255
	xi	0(%r1), 42
	xi	0(%r15), 42
	xi	4095(%r1), 42
	xi	4095(%r15), 42
