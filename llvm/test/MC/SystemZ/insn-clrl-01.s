# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clrl	%r0, -0x100000000
#CHECK: clrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clrl	%r0, -2
#CHECK: clrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clrl	%r0, 0
#CHECK: clrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clrl	%r0, 0xfffffffe

#CHECK: clrl	%r0, foo                # encoding: [0xc6,0x0f,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r15, foo               # encoding: [0xc6,0xff,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clrl	%r0,foo
	clrl	%r15,foo

#CHECK: clrl	%r3, bar+100            # encoding: [0xc6,0x3f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r4, bar+100            # encoding: [0xc6,0x4f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clrl	%r3,bar+100
	clrl	%r4,bar+100

#CHECK: clrl	%r7, frob@PLT           # encoding: [0xc6,0x7f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r8, frob@PLT           # encoding: [0xc6,0x8f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clrl	%r7,frob@PLT
	clrl	%r8,frob@PLT
