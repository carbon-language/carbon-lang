# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: strl	%r0, 2864434397         # encoding: [0xc4,0x0f,0x55,0x5d,0xe6,0x6e]
#CHECK: strl	%r15, 2864434397        # encoding: [0xc4,0xff,0x55,0x5d,0xe6,0x6e]

	strl	%r0,0xaabbccdd
	strl	%r15,0xaabbccdd

#CHECK: strl	%r0, foo                # encoding: [0xc4,0x0f,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: strl	%r15, foo               # encoding: [0xc4,0xff,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	strl	%r0,foo
	strl	%r15,foo

#CHECK: strl	%r3, bar+100            # encoding: [0xc4,0x3f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: strl	%r4, bar+100            # encoding: [0xc4,0x4f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	strl	%r3,bar+100
	strl	%r4,bar+100

#CHECK: strl	%r7, frob@PLT           # encoding: [0xc4,0x7f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: strl	%r8, frob@PLT           # encoding: [0xc4,0x8f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	strl	%r7,frob@PLT
	strl	%r8,frob@PLT
