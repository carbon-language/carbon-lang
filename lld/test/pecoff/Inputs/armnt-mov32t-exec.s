
# void function(void) { }
# void *get_function() { return &function; }

	.syntax unified
	.thumb
	.text

	.def function
		.scl 2
		.type 32
	.endef
	.global function
	.align 2
	.thumb_func
function:
	bx lr

	.def get_function
		.scl 2
		.type 32
	.endef
	.global get_function
	.align 2
	.thumb_func
get_function:
	movw r0, :lower16:function
	movt r0, :upper16:function
	bx lr

