# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lhrl	%r0, 2864434397         # encoding: [0xc4,0x05,0x55,0x5d,0xe6,0x6e]
#CHECK: lhrl	%r15, 2864434397        # encoding: [0xc4,0xf5,0x55,0x5d,0xe6,0x6e]

	lhrl	%r0,0xaabbccdd
	lhrl	%r15,0xaabbccdd

#CHECK: lhrl	%r0, foo                # encoding: [0xc4,0x05,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r15, foo               # encoding: [0xc4,0xf5,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lhrl	%r0,foo
	lhrl	%r15,foo

#CHECK: lhrl	%r3, bar+100            # encoding: [0xc4,0x35,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r4, bar+100            # encoding: [0xc4,0x45,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lhrl	%r3,bar+100
	lhrl	%r4,bar+100

#CHECK: lhrl	%r7, frob@PLT           # encoding: [0xc4,0x75,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r8, frob@PLT           # encoding: [0xc4,0x85,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lhrl	%r7,frob@PLT
	lhrl	%r8,frob@PLT
