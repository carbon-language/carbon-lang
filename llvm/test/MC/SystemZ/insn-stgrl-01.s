# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: stgrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	stgrl	%r0, -0x100000000
#CHECK: stgrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	stgrl	%r0, -2
#CHECK: stgrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	stgrl	%r0, 0
#CHECK: stgrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	stgrl	%r0, 0xfffffffe

#CHECK: stgrl	%r0, foo                # encoding: [0xc4,0x0b,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: stgrl	%r15, foo               # encoding: [0xc4,0xfb,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	stgrl	%r0,foo
	stgrl	%r15,foo

#CHECK: stgrl	%r3, bar+100            # encoding: [0xc4,0x3b,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: stgrl	%r4, bar+100            # encoding: [0xc4,0x4b,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	stgrl	%r3,bar+100
	stgrl	%r4,bar+100

#CHECK: stgrl	%r7, frob@PLT           # encoding: [0xc4,0x7b,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: stgrl	%r8, frob@PLT           # encoding: [0xc4,0x8b,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	stgrl	%r7,frob@PLT
	stgrl	%r8,frob@PLT
