; Testcase distilled from 256.bzip2.

target endian = little
target pointersize = 32

int %main() {
entry:
	br label %loopentry.0

loopentry.0:
	%h.0 = phi int [ %tmp.2, %loopentry.0 ], [ -1, %entry ]
	%tmp.2 = add int %h.0, 1
	%tmp.4 = setne int %tmp.2, 0
	br bool %tmp.4, label %loopentry.0, label %loopentry.1

loopentry.1:
	%h.1 = phi int [ %tmp.2, %loopentry.0 ]
	ret int %h.1
}
