
# __declspec(dllexport) void function(void) { }
# const void * const fps[] = { &function, };

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

	.section .rdata,"rd"
	.global fps
	.align 2
fps:
	.long function

