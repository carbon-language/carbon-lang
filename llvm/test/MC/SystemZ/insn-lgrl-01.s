# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgrl	%r0, 2864434397         # encoding: [0xc4,0x08,0x55,0x5d,0xe6,0x6e]
#CHECK: lgrl	%r15, 2864434397        # encoding: [0xc4,0xf8,0x55,0x5d,0xe6,0x6e]

	lgrl	%r0,0xaabbccdd
	lgrl	%r15,0xaabbccdd

#CHECK: lgrl	%r0, foo                # encoding: [0xc4,0x08,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r15, foo               # encoding: [0xc4,0xf8,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lgrl	%r0,foo
	lgrl	%r15,foo

#CHECK: lgrl	%r3, bar+100            # encoding: [0xc4,0x38,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r4, bar+100            # encoding: [0xc4,0x48,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lgrl	%r3,bar+100
	lgrl	%r4,bar+100

#CHECK: lgrl	%r7, frob@PLT           # encoding: [0xc4,0x78,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r8, frob@PLT           # encoding: [0xc4,0x88,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lgrl	%r7,frob@PLT
	lgrl	%r8,frob@PLT
