# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: sthrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	sthrl	%r0, -0x100000000
#CHECK: sthrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	sthrl	%r0, -2
#CHECK: sthrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	sthrl	%r0, 0
#CHECK: sthrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	sthrl	%r0, 0xfffffffe

#CHECK: sthrl	%r0, foo                # encoding: [0xc4,0x07,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: sthrl	%r15, foo               # encoding: [0xc4,0xf7,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	sthrl	%r0,foo
	sthrl	%r15,foo

#CHECK: sthrl	%r3, bar+100            # encoding: [0xc4,0x37,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: sthrl	%r4, bar+100            # encoding: [0xc4,0x47,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	sthrl	%r3,bar+100
	sthrl	%r4,bar+100

#CHECK: sthrl	%r7, frob@PLT           # encoding: [0xc4,0x77,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: sthrl	%r8, frob@PLT           # encoding: [0xc4,0x87,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	sthrl	%r7,frob@PLT
	sthrl	%r8,frob@PLT
