# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lghrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lghrl	%r0, -0x100000000
#CHECK: lghrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lghrl	%r0, -2
#CHECK: lghrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lghrl	%r0, 0
#CHECK: lghrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lghrl	%r0, 0xfffffffe

#CHECK: lghrl	%r0, foo                # encoding: [0xc4,0x04,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r15, foo               # encoding: [0xc4,0xf4,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lghrl	%r0,foo
	lghrl	%r15,foo

#CHECK: lghrl	%r3, bar+100            # encoding: [0xc4,0x34,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r4, bar+100            # encoding: [0xc4,0x44,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lghrl	%r3,bar+100
	lghrl	%r4,bar+100

#CHECK: lghrl	%r7, frob@PLT           # encoding: [0xc4,0x74,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r8, frob@PLT           # encoding: [0xc4,0x84,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lghrl	%r7,frob@PLT
	lghrl	%r8,frob@PLT
