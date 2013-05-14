# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lhrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lhrl	%r0, -0x100000000
#CHECK: lhrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lhrl	%r0, -2
#CHECK: lhrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lhrl	%r0, 0
#CHECK: lhrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lhrl	%r0, 0xfffffffe

#CHECK: lhrl	%r0, foo                # encoding: [0xc4,0x05,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r15, foo               # encoding: [0xc4,0xf5,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lhrl	%r0,foo
	lhrl	%r15,foo

#CHECK: lhrl	%r3, bar+100            # encoding: [0xc4,0x35,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r4, bar+100            # encoding: [0xc4,0x45,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lhrl	%r3,bar+100
	lhrl	%r4,bar+100

#CHECK: lhrl	%r7, frob@PLT           # encoding: [0xc4,0x75,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r8, frob@PLT           # encoding: [0xc4,0x85,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lhrl	%r7,frob@PLT
	lhrl	%r8,frob@PLT
