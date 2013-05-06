# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cghrl	%r0, 2864434397         # encoding: [0xc6,0x04,0x55,0x5d,0xe6,0x6e]
#CHECK: cghrl	%r15, 2864434397        # encoding: [0xc6,0xf4,0x55,0x5d,0xe6,0x6e]

	cghrl	%r0,0xaabbccdd
	cghrl	%r15,0xaabbccdd

#CHECK: cghrl	%r0, foo                # encoding: [0xc6,0x04,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: cghrl	%r15, foo               # encoding: [0xc6,0xf4,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	cghrl	%r0,foo
	cghrl	%r15,foo

#CHECK: cghrl	%r3, bar+100            # encoding: [0xc6,0x34,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: cghrl	%r4, bar+100            # encoding: [0xc6,0x44,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	cghrl	%r3,bar+100
	cghrl	%r4,bar+100

#CHECK: cghrl	%r7, frob@PLT           # encoding: [0xc6,0x74,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: cghrl	%r8, frob@PLT           # encoding: [0xc6,0x84,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	cghrl	%r7,frob@PLT
	cghrl	%r8,frob@PLT
