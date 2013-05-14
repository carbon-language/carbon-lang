# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clhrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clhrl	%r0, -0x100000000
#CHECK: clhrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clhrl	%r0, -2
#CHECK: clhrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clhrl	%r0, 0
#CHECK: clhrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clhrl	%r0, 0xfffffffe

#CHECK: clhrl	%r0, foo                # encoding: [0xc6,0x07,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r15, foo               # encoding: [0xc6,0xf7,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clhrl	%r0,foo
	clhrl	%r15,foo

#CHECK: clhrl	%r3, bar+100            # encoding: [0xc6,0x37,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r4, bar+100            # encoding: [0xc6,0x47,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clhrl	%r3,bar+100
	clhrl	%r4,bar+100

#CHECK: clhrl	%r7, frob@PLT           # encoding: [0xc6,0x77,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r8, frob@PLT           # encoding: [0xc6,0x87,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clhrl	%r7,frob@PLT
	clhrl	%r8,frob@PLT
