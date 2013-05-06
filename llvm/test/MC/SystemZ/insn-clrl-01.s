# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clrl	%r0, 2864434397         # encoding: [0xc6,0x0f,0x55,0x5d,0xe6,0x6e]
#CHECK: clrl	%r15, 2864434397        # encoding: [0xc6,0xff,0x55,0x5d,0xe6,0x6e]

	clrl	%r0,0xaabbccdd
	clrl	%r15,0xaabbccdd

#CHECK: clrl	%r0, foo                # encoding: [0xc6,0x0f,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r15, foo               # encoding: [0xc6,0xff,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clrl	%r0,foo
	clrl	%r15,foo

#CHECK: clrl	%r3, bar+100            # encoding: [0xc6,0x3f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r4, bar+100            # encoding: [0xc6,0x4f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clrl	%r3,bar+100
	clrl	%r4,bar+100

#CHECK: clrl	%r7, frob@PLT           # encoding: [0xc6,0x7f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r8, frob@PLT           # encoding: [0xc6,0x8f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clrl	%r7,frob@PLT
	clrl	%r8,frob@PLT
