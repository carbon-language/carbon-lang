// RUN: not llvm-mc -triple i686-linux -o /dev/null %s

	.macro macro parameter=0
		.if \parameter
		.else
	.endm

	macro 1

