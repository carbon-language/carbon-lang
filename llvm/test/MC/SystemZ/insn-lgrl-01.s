# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lgrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lgrl	%r0, -0x100000000
#CHECK: lgrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lgrl	%r0, -2
#CHECK: lgrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lgrl	%r0, 0
#CHECK: lgrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lgrl	%r0, 0xfffffffe

#CHECK: lgrl	%r0, foo                # encoding: [0xc4,0x08,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r15, foo               # encoding: [0xc4,0xf8,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lgrl	%r0,foo
	lgrl	%r15,foo

#CHECK: lgrl	%r3, bar+100            # encoding: [0xc4,0x38,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r4, bar+100            # encoding: [0xc4,0x48,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lgrl	%r3,bar+100
	lgrl	%r4,bar+100

#CHECK: lgrl	%r7, frob@PLT           # encoding: [0xc4,0x78,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r8, frob@PLT           # encoding: [0xc4,0x88,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lgrl	%r7,frob@PLT
	lgrl	%r8,frob@PLT
