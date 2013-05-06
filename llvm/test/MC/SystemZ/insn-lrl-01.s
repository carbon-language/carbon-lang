# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lrl	%r0, 2864434397         # encoding: [0xc4,0x0d,0x55,0x5d,0xe6,0x6e]
#CHECK: lrl	%r15, 2864434397        # encoding: [0xc4,0xfd,0x55,0x5d,0xe6,0x6e]

	lrl	%r0,0xaabbccdd
	lrl	%r15,0xaabbccdd

#CHECK: lrl	%r0, foo                # encoding: [0xc4,0x0d,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r15, foo               # encoding: [0xc4,0xfd,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lrl	%r0,foo
	lrl	%r15,foo

#CHECK: lrl	%r3, bar+100            # encoding: [0xc4,0x3d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r4, bar+100            # encoding: [0xc4,0x4d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lrl	%r3,bar+100
	lrl	%r4,bar+100

#CHECK: lrl	%r7, frob@PLT           # encoding: [0xc4,0x7d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r8, frob@PLT           # encoding: [0xc4,0x8d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lrl	%r7,frob@PLT
	lrl	%r8,frob@PLT
