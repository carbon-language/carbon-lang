# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: chrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	chrl	%r0, -0x100000000
#CHECK: chrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	chrl	%r0, -2
#CHECK: chrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	chrl	%r0, 0
#CHECK: chrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	chrl	%r0, 0xfffffffe

#CHECK: chrl	%r0, foo                # encoding: [0xc6,0x05,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r15, foo               # encoding: [0xc6,0xf5,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	chrl	%r0,foo
	chrl	%r15,foo

#CHECK: chrl	%r3, bar+100            # encoding: [0xc6,0x35,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r4, bar+100            # encoding: [0xc6,0x45,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	chrl	%r3,bar+100
	chrl	%r4,bar+100

#CHECK: chrl	%r7, frob@PLT           # encoding: [0xc6,0x75,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r8, frob@PLT           # encoding: [0xc6,0x85,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	chrl	%r7,frob@PLT
	chrl	%r8,frob@PLT
