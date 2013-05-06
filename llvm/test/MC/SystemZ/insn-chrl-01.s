# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: chrl	%r0, 2864434397         # encoding: [0xc6,0x05,0x55,0x5d,0xe6,0x6e]
#CHECK: chrl	%r15, 2864434397        # encoding: [0xc6,0xf5,0x55,0x5d,0xe6,0x6e]

	chrl	%r0,0xaabbccdd
	chrl	%r15,0xaabbccdd

#CHECK: chrl	%r0, foo                # encoding: [0xc6,0x05,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r15, foo               # encoding: [0xc6,0xf5,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	chrl	%r0,foo
	chrl	%r15,foo

#CHECK: chrl	%r3, bar+100            # encoding: [0xc6,0x35,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r4, bar+100            # encoding: [0xc6,0x45,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	chrl	%r3,bar+100
	chrl	%r4,bar+100

#CHECK: chrl	%r7, frob@PLT           # encoding: [0xc6,0x75,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r8, frob@PLT           # encoding: [0xc6,0x85,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	chrl	%r7,frob@PLT
	chrl	%r8,frob@PLT
