# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clgfrl	%r0, 2864434397         # encoding: [0xc6,0x0e,0x55,0x5d,0xe6,0x6e]
#CHECK: clgfrl	%r15, 2864434397        # encoding: [0xc6,0xfe,0x55,0x5d,0xe6,0x6e]

	clgfrl	%r0,0xaabbccdd
	clgfrl	%r15,0xaabbccdd

#CHECK: clgfrl	%r0, foo                # encoding: [0xc6,0x0e,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r15, foo               # encoding: [0xc6,0xfe,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clgfrl	%r0,foo
	clgfrl	%r15,foo

#CHECK: clgfrl	%r3, bar+100            # encoding: [0xc6,0x3e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r4, bar+100            # encoding: [0xc6,0x4e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clgfrl	%r3,bar+100
	clgfrl	%r4,bar+100

#CHECK: clgfrl	%r7, frob@PLT           # encoding: [0xc6,0x7e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r8, frob@PLT           # encoding: [0xc6,0x8e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clgfrl	%r7,frob@PLT
	clgfrl	%r8,frob@PLT
