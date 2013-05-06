# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llhrl	%r0, 2864434397         # encoding: [0xc4,0x02,0x55,0x5d,0xe6,0x6e]
#CHECK: llhrl	%r15, 2864434397        # encoding: [0xc4,0xf2,0x55,0x5d,0xe6,0x6e]

	llhrl	%r0,0xaabbccdd
	llhrl	%r15,0xaabbccdd

#CHECK: llhrl	%r0, foo                # encoding: [0xc4,0x02,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: llhrl	%r15, foo               # encoding: [0xc4,0xf2,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	llhrl	%r0,foo
	llhrl	%r15,foo

#CHECK: llhrl	%r3, bar+100            # encoding: [0xc4,0x32,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: llhrl	%r4, bar+100            # encoding: [0xc4,0x42,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	llhrl	%r3,bar+100
	llhrl	%r4,bar+100

#CHECK: llhrl	%r7, frob@PLT           # encoding: [0xc4,0x72,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: llhrl	%r8, frob@PLT           # encoding: [0xc4,0x82,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	llhrl	%r7,frob@PLT
	llhrl	%r8,frob@PLT
