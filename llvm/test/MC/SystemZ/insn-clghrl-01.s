# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clghrl	%r0, 2864434397         # encoding: [0xc6,0x06,0x55,0x5d,0xe6,0x6e]
#CHECK: clghrl	%r15, 2864434397        # encoding: [0xc6,0xf6,0x55,0x5d,0xe6,0x6e]

	clghrl	%r0,0xaabbccdd
	clghrl	%r15,0xaabbccdd

#CHECK: clghrl	%r0, foo                # encoding: [0xc6,0x06,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r15, foo               # encoding: [0xc6,0xf6,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clghrl	%r0,foo
	clghrl	%r15,foo

#CHECK: clghrl	%r3, bar+100            # encoding: [0xc6,0x36,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r4, bar+100            # encoding: [0xc6,0x46,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clghrl	%r3,bar+100
	clghrl	%r4,bar+100

#CHECK: clghrl	%r7, frob@PLT           # encoding: [0xc6,0x76,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r8, frob@PLT           # encoding: [0xc6,0x86,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clghrl	%r7,frob@PLT
	clghrl	%r8,frob@PLT
