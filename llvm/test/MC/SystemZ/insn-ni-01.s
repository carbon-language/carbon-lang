# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: ni	0, 0                    # encoding: [0x94,0x00,0x00,0x00]
#CHECK: ni	4095, 0                 # encoding: [0x94,0x00,0x0f,0xff]
#CHECK: ni	0, 255                  # encoding: [0x94,0xff,0x00,0x00]
#CHECK: ni	0(%r1), 42              # encoding: [0x94,0x2a,0x10,0x00]
#CHECK: ni	0(%r15), 42             # encoding: [0x94,0x2a,0xf0,0x00]
#CHECK: ni	4095(%r1), 42           # encoding: [0x94,0x2a,0x1f,0xff]
#CHECK: ni	4095(%r15), 42          # encoding: [0x94,0x2a,0xff,0xff]

	ni	0, 0
	ni	4095, 0
	ni	0, 255
	ni	0(%r1), 42
	ni	0(%r15), 42
	ni	4095(%r1), 42
	ni	4095(%r15), 42
