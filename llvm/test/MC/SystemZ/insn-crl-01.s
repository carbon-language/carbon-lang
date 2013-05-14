# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: crl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	crl	%r0, -0x100000000
#CHECK: crl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	crl	%r0, -2
#CHECK: crl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	crl	%r0, 0
#CHECK: crl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	crl	%r0, 0xfffffffe

#CHECK: crl	%r0, foo                # encoding: [0xc6,0x0d,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: crl	%r15, foo               # encoding: [0xc6,0xfd,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	crl	%r0,foo
	crl	%r15,foo

#CHECK: crl	%r3, bar+100            # encoding: [0xc6,0x3d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: crl	%r4, bar+100            # encoding: [0xc6,0x4d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	crl	%r3,bar+100
	crl	%r4,bar+100

#CHECK: crl	%r7, frob@PLT           # encoding: [0xc6,0x7d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: crl	%r8, frob@PLT           # encoding: [0xc6,0x8d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	crl	%r7,frob@PLT
	crl	%r8,frob@PLT
