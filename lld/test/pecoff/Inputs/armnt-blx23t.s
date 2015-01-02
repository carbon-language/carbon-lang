
@ __declspec(noinline) int identity(int i) { return i; }
@ int function() { return identity(32) + 1; }

	.syntax unified
	.thumb
	.text

	.def identity
		.scl 2
		.type 32
	.endef
	.global identity
	.align 2
	.code16
	.thumb_func
identity:
	bx lr

	.def function
		.scl 2
		.type 32
	.endef
	.global function
	.align 2
	.code16
	.thumb_func
function:
	push.w {r11, lr}
	mov r11, sp
	movs r0, #32
	bl identity
	adds r0, #1
	pop.w {r11, pc}

