@ RUN: llvm-mc -triple thumbv7-unknown-unknown -show-encoding %s > %t
@ RUN: FileCheck < %t %s

	.syntax unified
	.text

@ FIXME: This is not the correct instruction representation, but at least we are
@ parsing the ldr to something.
@
@ CHECK: ldr r0, [r7, #258]
	ldr	r0, [r7, #-8]
        
