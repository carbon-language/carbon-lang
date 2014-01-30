@ RUN: not llvm-mc -triple armv7-linux-gnu -filetype asm -o /dev/null %s 2>&1 \
@ RUN:   | FileCheck %s

	.type missing_variable,%function
missing_variable:
.tlsdescseq

@ CHECK: error: expected variable after '.tlsdescseq' directive
@ CHECK: 	.tlsdescseq
@ CHECK:                   ^

	.type bad_expression,%function
bad_expression:
.tlsdescseq variable(tlsdesc)

@ CHECK: error: unexpected token
@ CHECK: 	.tlsdescseq variable(tlsdesc)
@ CHECK:                            ^

	.type trailing_garbage,%function
trailing_garbage:
.tlsdescseq variable,

@ CHECK: error: unexpected token
@ CHECK: 	.tlsdescseq variable,
@ CHECK:                            ^

	.type invalid_use,%function
invalid_use:
	blx invalid(tlsdescseq)

@ CHECK: error: invalid variant 'tlsdescseq'
@ CHECK: 	blx invalid(tlsdescseq)
@ CHECK:                    ^

