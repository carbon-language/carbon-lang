# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	llgfrl	%r0, -0x100000000
#CHECK: llgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	llgfrl	%r0, -2
#CHECK: llgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	llgfrl	%r0, 0
#CHECK: llgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	llgfrl	%r0, 0xfffffffe

#CHECK: llgfrl	%r0, foo                # encoding: [0xc4,0x0e,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r15, foo               # encoding: [0xc4,0xfe,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	llgfrl	%r0,foo
	llgfrl	%r15,foo

#CHECK: llgfrl	%r3, bar+100            # encoding: [0xc4,0x3e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r4, bar+100            # encoding: [0xc4,0x4e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	llgfrl	%r3,bar+100
	llgfrl	%r4,bar+100

#CHECK: llgfrl	%r7, frob@PLT           # encoding: [0xc4,0x7e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r8, frob@PLT           # encoding: [0xc4,0x8e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	llgfrl	%r7,frob@PLT
	llgfrl	%r8,frob@PLT
