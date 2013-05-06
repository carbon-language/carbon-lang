# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgfrl	%r0, 2864434397         # encoding: [0xc6,0x0c,0x55,0x5d,0xe6,0x6e]
#CHECK: cgfrl	%r15, 2864434397        # encoding: [0xc6,0xfc,0x55,0x5d,0xe6,0x6e]

	cgfrl	%r0,0xaabbccdd
	cgfrl	%r15,0xaabbccdd

#CHECK: cgfrl	%r0, foo                # encoding: [0xc6,0x0c,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r15, foo               # encoding: [0xc6,0xfc,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	cgfrl	%r0,foo
	cgfrl	%r15,foo

#CHECK: cgfrl	%r3, bar+100            # encoding: [0xc6,0x3c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r4, bar+100            # encoding: [0xc6,0x4c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	cgfrl	%r3,bar+100
	cgfrl	%r4,bar+100

#CHECK: cgfrl	%r7, frob@PLT           # encoding: [0xc6,0x7c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r8, frob@PLT           # encoding: [0xc6,0x8c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	cgfrl	%r7,frob@PLT
	cgfrl	%r8,frob@PLT
