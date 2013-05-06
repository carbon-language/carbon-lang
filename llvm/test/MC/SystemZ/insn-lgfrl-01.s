# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgfrl	%r0, 2864434397         # encoding: [0xc4,0x0c,0x55,0x5d,0xe6,0x6e]
#CHECK: lgfrl	%r15, 2864434397        # encoding: [0xc4,0xfc,0x55,0x5d,0xe6,0x6e]

	lgfrl	%r0,0xaabbccdd
	lgfrl	%r15,0xaabbccdd

#CHECK: lgfrl	%r0, foo                # encoding: [0xc4,0x0c,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r15, foo               # encoding: [0xc4,0xfc,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lgfrl	%r0,foo
	lgfrl	%r15,foo

#CHECK: lgfrl	%r3, bar+100            # encoding: [0xc4,0x3c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r4, bar+100            # encoding: [0xc4,0x4c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lgfrl	%r3,bar+100
	lgfrl	%r4,bar+100

#CHECK: lgfrl	%r7, frob@PLT           # encoding: [0xc4,0x7c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r8, frob@PLT           # encoding: [0xc4,0x8c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lgfrl	%r7,frob@PLT
	lgfrl	%r8,frob@PLT
