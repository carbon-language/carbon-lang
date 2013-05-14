# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lrl	%r0, -0x100000000
#CHECK: lrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lrl	%r0, -2
#CHECK: lrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lrl	%r0, 0
#CHECK: lrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lrl	%r0, 0xfffffffe

#CHECK: lrl	%r0, foo                # encoding: [0xc4,0x0d,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r15, foo               # encoding: [0xc4,0xfd,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lrl	%r0,foo
	lrl	%r15,foo

#CHECK: lrl	%r3, bar+100            # encoding: [0xc4,0x3d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r4, bar+100            # encoding: [0xc4,0x4d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lrl	%r3,bar+100
	lrl	%r4,bar+100

#CHECK: lrl	%r7, frob@PLT           # encoding: [0xc4,0x7d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r8, frob@PLT           # encoding: [0xc4,0x8d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lrl	%r7,frob@PLT
	lrl	%r8,frob@PLT
