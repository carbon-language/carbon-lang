# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lghrl	%r0, 2864434397         # encoding: [0xc4,0x04,0x55,0x5d,0xe6,0x6e]
#CHECK: lghrl	%r15, 2864434397        # encoding: [0xc4,0xf4,0x55,0x5d,0xe6,0x6e]

	lghrl	%r0,0xaabbccdd
	lghrl	%r15,0xaabbccdd

#CHECK: lghrl	%r0, foo                # encoding: [0xc4,0x04,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r15, foo               # encoding: [0xc4,0xf4,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lghrl	%r0,foo
	lghrl	%r15,foo

#CHECK: lghrl	%r3, bar+100            # encoding: [0xc4,0x34,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r4, bar+100            # encoding: [0xc4,0x44,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lghrl	%r3,bar+100
	lghrl	%r4,bar+100

#CHECK: lghrl	%r7, frob@PLT           # encoding: [0xc4,0x74,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r8, frob@PLT           # encoding: [0xc4,0x84,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lghrl	%r7,frob@PLT
	lghrl	%r8,frob@PLT
