# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llgfrl	%r0, 2864434397         # encoding: [0xc4,0x0e,0x55,0x5d,0xe6,0x6e]
#CHECK: llgfrl	%r15, 2864434397        # encoding: [0xc4,0xfe,0x55,0x5d,0xe6,0x6e]

	llgfrl	%r0,0xaabbccdd
	llgfrl	%r15,0xaabbccdd

#CHECK: llgfrl	%r0, foo                # encoding: [0xc4,0x0e,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r15, foo               # encoding: [0xc4,0xfe,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	llgfrl	%r0,foo
	llgfrl	%r15,foo

#CHECK: llgfrl	%r3, bar+100            # encoding: [0xc4,0x3e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r4, bar+100            # encoding: [0xc4,0x4e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	llgfrl	%r3,bar+100
	llgfrl	%r4,bar+100

#CHECK: llgfrl	%r7, frob@PLT           # encoding: [0xc4,0x7e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r8, frob@PLT           # encoding: [0xc4,0x8e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	llgfrl	%r7,frob@PLT
	llgfrl	%r8,frob@PLT
