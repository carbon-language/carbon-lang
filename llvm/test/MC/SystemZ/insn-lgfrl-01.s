# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lgfrl	%r0, -0x100000000
#CHECK: lgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lgfrl	%r0, -2
#CHECK: lgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lgfrl	%r0, 0
#CHECK: lgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lgfrl	%r0, 0xfffffffe

#CHECK: lgfrl	%r0, foo                # encoding: [0xc4,0x0c,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r15, foo               # encoding: [0xc4,0xfc,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lgfrl	%r0,foo
	lgfrl	%r15,foo

#CHECK: lgfrl	%r3, bar+100            # encoding: [0xc4,0x3c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r4, bar+100            # encoding: [0xc4,0x4c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lgfrl	%r3,bar+100
	lgfrl	%r4,bar+100

#CHECK: lgfrl	%r7, frob@PLT           # encoding: [0xc4,0x7c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r8, frob@PLT           # encoding: [0xc4,0x8c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lgfrl	%r7,frob@PLT
	lgfrl	%r8,frob@PLT
