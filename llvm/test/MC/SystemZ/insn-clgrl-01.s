# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clgrl	%r0, 2864434397         # encoding: [0xc6,0x0a,0x55,0x5d,0xe6,0x6e]
#CHECK: clgrl	%r15, 2864434397        # encoding: [0xc6,0xfa,0x55,0x5d,0xe6,0x6e]

	clgrl	%r0,0xaabbccdd
	clgrl	%r15,0xaabbccdd

#CHECK: clgrl	%r0, foo                # encoding: [0xc6,0x0a,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clgrl	%r15, foo               # encoding: [0xc6,0xfa,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clgrl	%r0,foo
	clgrl	%r15,foo

#CHECK: clgrl	%r3, bar+100            # encoding: [0xc6,0x3a,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clgrl	%r4, bar+100            # encoding: [0xc6,0x4a,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clgrl	%r3,bar+100
	clgrl	%r4,bar+100

#CHECK: clgrl	%r7, frob@PLT           # encoding: [0xc6,0x7a,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clgrl	%r8, frob@PLT           # encoding: [0xc6,0x8a,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clgrl	%r7,frob@PLT
	clgrl	%r8,frob@PLT
