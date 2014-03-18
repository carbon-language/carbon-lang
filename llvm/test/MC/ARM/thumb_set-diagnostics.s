@ RUN: not llvm-mc -triple armv7-eabi -o /dev/null 2>&1 %s | FileCheck %s

	.syntax unified

	.thumb

	.thumb_set

@ CHECK: error: expected identifier after '.thumb_set'
@ CHECK: 	.thumb_set
@ CHECL:                  ^

	.thumb_set ., 0x0b5e55ed

@ CHECK: error: expected identifier after '.thumb_set'
@ CHECK: 	.thumb_set ., 0x0b5e55ed
@ CHECK:                   ^

	.thumb_set labelled, 0x1abe11ed
	.thumb_set invalid, :lower16:labelled

@ CHECK: error: unknown token in expression
@ CHECK: 	.thumb_set invalid, :lower16:labelled
@ CHECK:                            ^

	.thumb_set missing_comma

@ CHECK: error: expected comma after name 'missing_comma'
@ CHECK: 	.thumb_set missing_comma
@ CHECK:                                ^

	.thumb_set missing_expression,

@ CHECK: error: missing expression
@ CHECK: 	.thumb_set missing_expression,
@ CHECK:                                      ^

	.thumb_set trailer_trash, 0x11fe1e55,

@ CHECK: error: unexpected token
@ CHECK: 	.thumb_set trailer_trash, 0x11fe1e55,
@ CHECK:                                            ^

