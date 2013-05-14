# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: larl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	larl	%r0, -0x100000000
#CHECK: larl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	larl	%r0, -2
#CHECK: larl	%r0, .[[LAB:L.*]]	# encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	larl	%r0, 0
#CHECK: larl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	larl	%r0, 0xfffffffe

#CHECK: larl	%r0, foo                # encoding: [0xc0,0x00,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: larl	%r15, foo               # encoding: [0xc0,0xf0,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	larl	%r0,foo
	larl	%r15,foo

#CHECK: larl	%r3, bar+100            # encoding: [0xc0,0x30,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: larl	%r4, bar+100            # encoding: [0xc0,0x40,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	larl	%r3,bar+100
	larl	%r4,bar+100

#CHECK: larl	%r7, frob@PLT           # encoding: [0xc0,0x70,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: larl	%r8, frob@PLT           # encoding: [0xc0,0x80,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	larl	%r7,frob@PLT
	larl	%r8,frob@PLT
