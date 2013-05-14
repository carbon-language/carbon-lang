# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: cgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	cgfrl	%r0, -0x100000000
#CHECK: cgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	cgfrl	%r0, -2
#CHECK: cgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	cgfrl	%r0, 0
#CHECK: cgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	cgfrl	%r0, 0xfffffffe

#CHECK: cgfrl	%r0, foo                # encoding: [0xc6,0x0c,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r15, foo               # encoding: [0xc6,0xfc,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	cgfrl	%r0,foo
	cgfrl	%r15,foo

#CHECK: cgfrl	%r3, bar+100            # encoding: [0xc6,0x3c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r4, bar+100            # encoding: [0xc6,0x4c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	cgfrl	%r3,bar+100
	cgfrl	%r4,bar+100

#CHECK: cgfrl	%r7, frob@PLT           # encoding: [0xc6,0x7c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r8, frob@PLT           # encoding: [0xc6,0x8c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	cgfrl	%r7,frob@PLT
	cgfrl	%r8,frob@PLT
