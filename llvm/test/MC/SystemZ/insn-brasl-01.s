# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: brasl	%r0, foo                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, foo               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, foo               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brasl	%r0,foo
	brasl	%r14,foo
	brasl	%r15,foo

#CHECK: brasl	%r0, bar+100                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, bar+100               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, bar+100               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	brasl	%r0,bar+100
	brasl	%r14,bar+100
	brasl	%r15,bar+100

#CHECK: brasl	%r0, bar@PLT                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, bar@PLT               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, bar@PLT               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	brasl	%r0,bar@PLT
	brasl	%r14,bar@PLT
	brasl	%r15,bar@PLT
