# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clgfrl	%r0, -0x100000000
#CHECK: clgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clgfrl	%r0, -2
#CHECK: clgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clgfrl	%r0, 0
#CHECK: clgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clgfrl	%r0, 0xfffffffe

#CHECK: clgfrl	%r0, foo                # encoding: [0xc6,0x0e,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r15, foo               # encoding: [0xc6,0xfe,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clgfrl	%r0,foo
	clgfrl	%r15,foo

#CHECK: clgfrl	%r3, bar+100            # encoding: [0xc6,0x3e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r4, bar+100            # encoding: [0xc6,0x4e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clgfrl	%r3,bar+100
	clgfrl	%r4,bar+100

#CHECK: clgfrl	%r7, frob@PLT           # encoding: [0xc6,0x7e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r8, frob@PLT           # encoding: [0xc6,0x8e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clgfrl	%r7,frob@PLT
	clgfrl	%r8,frob@PLT
