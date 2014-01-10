@ RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s | FileCheck %s

	.syntax unified

	.type function,%function
function:
	bx lr

.set deadbeat, 0xdeadbea7

	.type test,%function
test:
	movw r0, :lower16:function
	movt r0, :upper16:function

	movw r1, #:lower16:function
	movt r1, #:upper16:function

	movw r2, :lower16:deadbeat
	movt r2, :upper16:deadbeat

	movw r3, #:lower16:deadbeat
	movt r3, #:upper16:deadbeat

	movw r4, :lower16:0xD1510D6E
	movt r4, :upper16:0xD1510D6E

	movw r5, #:lower16:0xD1510D6E
	movt r5, #:upper16:0xD1510D6E

@ CHECK-LABEL: test:
@ CHECK: 	movw r0, :lower16:function
@ CHECK: 	movt r0, :upper16:function
@ CHECK: 	movw r1, :lower16:function
@ CHECK: 	movt r1, :upper16:function
@ CHECK: 	movw r2, :lower16:(3735928487)
@ CHECK: 	movt r2, :upper16:(3735928487)
@ CHECK: 	movw r3, :lower16:(3735928487)
@ CHECK: 	movt r3, :upper16:(3735928487)
@ CHECK: 	movw r4, :lower16:(3511749998)
@ CHECK: 	movt r4, :upper16:(3511749998)
@ CHECK: 	movw r5, :lower16:(3511749998)
@ CHECK: 	movt r5, :upper16:(3511749998)

