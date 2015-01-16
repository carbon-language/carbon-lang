
	.syntax unified
	.thumb
	.text

	.def mainCRTStartup
		.type 32
		.scl 2
	.endef
	.align 2
	.thumb_func
mainCRTStartup:
	bx lr
	trap
	.long __ImageBase

