
@ int ___declspec(noinline) identity(int i) { return i; }
@ int function(void) { return identity(32); }

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
function:
	movs r0, #32
	b identity

