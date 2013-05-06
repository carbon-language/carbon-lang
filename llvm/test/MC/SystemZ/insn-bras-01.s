# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: bras	%r0, foo                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, foo               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, foo               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	bras	%r0,foo
	bras	%r14,foo
	bras	%r15,foo

#CHECK: bras	%r0, bar+100                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, bar+100               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, bar+100               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	bras	%r0,bar+100
	bras	%r14,bar+100
	bras	%r15,bar+100

#CHECK: bras	%r0, bar@PLT                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, bar@PLT               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, bar@PLT               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	bras	%r0,bar@PLT
	bras	%r14,bar@PLT
	bras	%r15,bar@PLT
