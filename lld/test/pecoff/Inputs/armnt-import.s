
# void __declspec(dllimport) function(void);
# int mainCRTStartup(void) { return function(); }

	.syntax unified
	.thumb
	.text

	.def mainCRTStartup
		.scl 2
		.type 32
	.endef
	.global mainCRTStartup
	.align 2
	.thumb_func
mainCRTStartup:
	movw r0, :lower16:__imp_function
	movt r0, :upper16:__imp_function
	ldr r0, [r0]
	bx r0

