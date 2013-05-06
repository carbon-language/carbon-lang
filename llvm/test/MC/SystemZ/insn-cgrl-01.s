# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgrl	%r0, 2864434397         # encoding: [0xc6,0x08,0x55,0x5d,0xe6,0x6e]
#CHECK: cgrl	%r15, 2864434397        # encoding: [0xc6,0xf8,0x55,0x5d,0xe6,0x6e]

	cgrl	%r0,0xaabbccdd
	cgrl	%r15,0xaabbccdd

#CHECK: cgrl	%r0, foo                # encoding: [0xc6,0x08,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: cgrl	%r15, foo               # encoding: [0xc6,0xf8,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	cgrl	%r0,foo
	cgrl	%r15,foo

#CHECK: cgrl	%r3, bar+100            # encoding: [0xc6,0x38,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: cgrl	%r4, bar+100            # encoding: [0xc6,0x48,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	cgrl	%r3,bar+100
	cgrl	%r4,bar+100

#CHECK: cgrl	%r7, frob@PLT           # encoding: [0xc6,0x78,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: cgrl	%r8, frob@PLT           # encoding: [0xc6,0x88,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	cgrl	%r7,frob@PLT
	cgrl	%r8,frob@PLT
