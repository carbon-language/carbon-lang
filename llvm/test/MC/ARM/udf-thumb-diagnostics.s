@ RUN: not llvm-mc -triple thumbv6m-eabi %s 2>&1 | FileCheck %s

	.syntax unified
	.text
	.thumb

undefined:
	udfpl

@ CHECK: error: conditional execution not supported in Thumb1
@ CHECK: 	udfpl
@ CHECK: 	^

	udf #256

@ CHECK: error: instruction requires: arm-mode
@ CHECK: 	udf #256
@ CHECK: 	^

