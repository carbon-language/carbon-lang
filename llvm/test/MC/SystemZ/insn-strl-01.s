# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: strl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	strl	%r0, -0x100000000
#CHECK: strl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	strl	%r0, -2
#CHECK: strl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	strl	%r0, 0
#CHECK: strl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	strl	%r0, 0xfffffffe

#CHECK: strl	%r0, foo                # encoding: [0xc4,0x0f,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: strl	%r15, foo               # encoding: [0xc4,0xff,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	strl	%r0,foo
	strl	%r15,foo

#CHECK: strl	%r3, bar+100            # encoding: [0xc4,0x3f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: strl	%r4, bar+100            # encoding: [0xc4,0x4f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	strl	%r3,bar+100
	strl	%r4,bar+100

#CHECK: strl	%r7, frob@PLT           # encoding: [0xc4,0x7f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: strl	%r8, frob@PLT           # encoding: [0xc4,0x8f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	strl	%r7,frob@PLT
	strl	%r8,frob@PLT
