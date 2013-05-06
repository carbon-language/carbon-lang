# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clhrl	%r0, 2864434397         # encoding: [0xc6,0x07,0x55,0x5d,0xe6,0x6e]
#CHECK: clhrl	%r15, 2864434397        # encoding: [0xc6,0xf7,0x55,0x5d,0xe6,0x6e]

	clhrl	%r0,0xaabbccdd
	clhrl	%r15,0xaabbccdd

#CHECK: clhrl	%r0, foo                # encoding: [0xc6,0x07,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r15, foo               # encoding: [0xc6,0xf7,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clhrl	%r0,foo
	clhrl	%r15,foo

#CHECK: clhrl	%r3, bar+100            # encoding: [0xc6,0x37,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r4, bar+100            # encoding: [0xc6,0x47,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clhrl	%r3,bar+100
	clhrl	%r4,bar+100

#CHECK: clhrl	%r7, frob@PLT           # encoding: [0xc6,0x77,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r8, frob@PLT           # encoding: [0xc6,0x87,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clhrl	%r7,frob@PLT
	clhrl	%r8,frob@PLT
