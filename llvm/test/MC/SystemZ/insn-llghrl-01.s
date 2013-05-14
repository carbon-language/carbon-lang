# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: llghrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	llghrl	%r0, -0x100000000
#CHECK: llghrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	llghrl	%r0, -2
#CHECK: llghrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	llghrl	%r0, 0
#CHECK: llghrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	llghrl	%r0, 0xfffffffe

#CHECK: llghrl	%r0, foo                # encoding: [0xc4,0x06,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: llghrl	%r15, foo               # encoding: [0xc4,0xf6,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	llghrl	%r0,foo
	llghrl	%r15,foo

#CHECK: llghrl	%r3, bar+100            # encoding: [0xc4,0x36,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: llghrl	%r4, bar+100            # encoding: [0xc4,0x46,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	llghrl	%r3,bar+100
	llghrl	%r4,bar+100

#CHECK: llghrl	%r7, frob@PLT           # encoding: [0xc4,0x76,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: llghrl	%r8, frob@PLT           # encoding: [0xc4,0x86,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	llghrl	%r7,frob@PLT
	llghrl	%r8,frob@PLT
