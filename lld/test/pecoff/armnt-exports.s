
# void __declspec(dllexport) function() {}
# void _DllMainCRTStartup() {}

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

	.def _DllMainCRTStartup
		.scl 2
		.type 32
	.endef
	.global _DllMainCRTStartup
	.align 2
	.thumb_func
_DllMainCRTStartup
	bx lr

