
# static const char buffer[] = "buffer";
# const char *get_buffer() { return buffer; }

	.syntax unified
	.thumb
	.text

	.def get_buffer
		.scl 2
		.type 32
	.endef
	.global get_buffer
	.align 2
	.thumb_func
get_buffer:
	movw r0, :lower16:buffer
	movt r0, :upper16:buffer
	bx lr

	.section .rdata,"rd"
buffer:
	.asciz "buffer"

