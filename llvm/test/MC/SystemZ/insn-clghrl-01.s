# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: clghrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clghrl	%r0, -0x100000000
#CHECK: clghrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clghrl	%r0, -2
#CHECK: clghrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clghrl	%r0, 0
#CHECK: clghrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clghrl	%r0, 0xfffffffe

#CHECK: clghrl	%r0, foo                # encoding: [0xc6,0x06,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r15, foo               # encoding: [0xc6,0xf6,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clghrl	%r0,foo
	clghrl	%r15,foo

#CHECK: clghrl	%r3, bar+100            # encoding: [0xc6,0x36,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r4, bar+100            # encoding: [0xc6,0x46,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clghrl	%r3,bar+100
	clghrl	%r4,bar+100

#CHECK: clghrl	%r7, frob@PLT           # encoding: [0xc6,0x76,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r8, frob@PLT           # encoding: [0xc6,0x86,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clghrl	%r7,frob@PLT
	clghrl	%r8,frob@PLT
