# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: brcl	0, foo                  # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	0, foo

#CHECK: brcl	1, foo                  # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgo	foo                     # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	1, foo
	jgo	foo

#CHECK: brcl	2, foo                  # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgh	foo                     # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	2, foo
	jgh	foo

#CHECK: brcl	3, foo                  # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnle	foo                     # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	3, foo
	jgnle	foo

#CHECK: brcl	4, foo                  # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgl	foo                     # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	4, foo
	jgl	foo

#CHECK: brcl	5, foo                  # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnhe	foo                     # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	5, foo
	jgnhe	foo

#CHECK: brcl	6, foo                  # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jglh	foo                     # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	6, foo
	jglh	foo

#CHECK: brcl	7, foo                  # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgne	foo                     # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	7, foo
	jgne	foo

#CHECK: brcl	8, foo                  # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jge	foo                     # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	8, foo
	jge	foo

#CHECK: brcl	9, foo                  # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnlh	foo                     # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	9, foo
	jgnlh	foo

#CHECK: brcl	10, foo                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jghe	foo                     # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	10, foo
	jghe	foo

#CHECK: brcl	11, foo                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnl	foo                     # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	11, foo
	jgnl	foo

#CHECK: brcl	12, foo                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgle	foo                     # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	12, foo
	jgle	foo

#CHECK: brcl	13, foo                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnh	foo                     # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	13, foo
	jgnh	foo

#CHECK: brcl	14, foo                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgno	foo                     # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	14, foo
	jgno	foo

#CHECK: brcl	15, foo                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jg	foo                     # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	15, foo
	jg	foo

#CHECK: brcl	0, bar+100              # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	brcl	0, bar+100

#CHECK: jgo	bar+100                 # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgo	bar+100

#CHECK: jgh	bar+100                 # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgh	bar+100

#CHECK: jgnle	bar+100                 # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnle	bar+100

#CHECK: jgl	bar+100                 # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgl	bar+100

#CHECK: jgnhe	bar+100                 # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnhe	bar+100

#CHECK: jglh	bar+100                 # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jglh	bar+100

#CHECK: jgne	bar+100                 # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgne	bar+100

#CHECK: jge	bar+100                 # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jge	bar+100

#CHECK: jgnlh	bar+100                 # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnlh	bar+100

#CHECK: jghe	bar+100                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jghe	bar+100

#CHECK: jgnl	bar+100                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnl	bar+100

#CHECK: jgle	bar+100                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgle	bar+100

#CHECK: jgnh	bar+100                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnh	bar+100

#CHECK: jgno	bar+100                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgno	bar+100

#CHECK: jg	bar+100                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jg	bar+100

#CHECK: brcl	0, bar@PLT              # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	brcl	0, bar@PLT

#CHECK: jgo	bar@PLT                 # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgo	bar@PLT

#CHECK: jgh	bar@PLT                 # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgh	bar@PLT

#CHECK: jgnle	bar@PLT                 # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnle	bar@PLT

#CHECK: jgl	bar@PLT                 # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgl	bar@PLT

#CHECK: jgnhe	bar@PLT                 # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnhe	bar@PLT

#CHECK: jglh	bar@PLT                 # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jglh	bar@PLT

#CHECK: jgne	bar@PLT                 # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgne	bar@PLT

#CHECK: jge	bar@PLT                 # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jge	bar@PLT

#CHECK: jgnlh	bar@PLT                 # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnlh	bar@PLT

#CHECK: jghe	bar@PLT                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jghe	bar@PLT

#CHECK: jgnl	bar@PLT                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnl	bar@PLT

#CHECK: jgle	bar@PLT                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgle	bar@PLT

#CHECK: jgnh	bar@PLT                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnh	bar@PLT

#CHECK: jgno	bar@PLT                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgno	bar@PLT

#CHECK: jg	bar@PLT                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jg	bar@PLT
