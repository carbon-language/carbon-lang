# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: crl	%r0, 2864434397         # encoding: [0xc6,0x0d,0x55,0x5d,0xe6,0x6e]
#CHECK: crl	%r15, 2864434397        # encoding: [0xc6,0xfd,0x55,0x5d,0xe6,0x6e]

	crl	%r0,0xaabbccdd
	crl	%r15,0xaabbccdd

#CHECK: crl	%r0, foo                # encoding: [0xc6,0x0d,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: crl	%r15, foo               # encoding: [0xc6,0xfd,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	crl	%r0,foo
	crl	%r15,foo

#CHECK: crl	%r3, bar+100            # encoding: [0xc6,0x3d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: crl	%r4, bar+100            # encoding: [0xc6,0x4d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	crl	%r3,bar+100
	crl	%r4,bar+100

#CHECK: crl	%r7, frob@PLT           # encoding: [0xc6,0x7d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: crl	%r8, frob@PLT           # encoding: [0xc6,0x8d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	crl	%r7,frob@PLT
	crl	%r8,frob@PLT
