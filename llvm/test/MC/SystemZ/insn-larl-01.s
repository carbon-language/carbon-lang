# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: larl	%r0, 2864434397         # encoding: [0xc0,0x00,0x55,0x5d,0xe6,0x6e]
#CHECK: larl	%r15, 2864434397        # encoding: [0xc0,0xf0,0x55,0x5d,0xe6,0x6e]

	larl	%r0,0xaabbccdd
	larl	%r15,0xaabbccdd

#CHECK: larl	%r0, foo                # encoding: [0xc0,0x00,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: larl	%r15, foo               # encoding: [0xc0,0xf0,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	larl	%r0,foo
	larl	%r15,foo

#CHECK: larl	%r3, bar+100            # encoding: [0xc0,0x30,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: larl	%r4, bar+100            # encoding: [0xc0,0x40,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	larl	%r3,bar+100
	larl	%r4,bar+100

#CHECK: larl	%r7, frob@PLT           # encoding: [0xc0,0x70,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: larl	%r8, frob@PLT           # encoding: [0xc0,0x80,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	larl	%r7,frob@PLT
	larl	%r8,frob@PLT
