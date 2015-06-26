	.section	__TEXT,__text,regular,pure_instructions
	.globl	_trivial_patchpoint_codegen
	.align	4, 0x90
_trivial_patchpoint_codegen:            ## @trivial_patchpoint_codegen
	.fill	1
Ltmp3:

	.section	__LLVM_STACKMAPS,__llvm_stackmaps
__LLVM_StackMaps:
	.byte	1
	.byte	0
	.short	0
	.long	1
	.long	1
	.long	1
	.quad	_trivial_patchpoint_codegen
	.quad	16
	.quad	10000000000
	.quad	2
	.long	Ltmp3-_trivial_patchpoint_codegen
	.short	0
	.short	5
	.byte	1
	.byte	8
	.short	5
	.long	0
	.byte	4
	.byte	8
	.short	0
	.long	10
	.byte	5
	.byte	8
	.short	0
	.long	0
	.byte	2
	.byte	8
	.short	4
	.long	-8
	.byte	3
	.byte	8
	.short	6
	.long	-16
	.short	0
	.short	1
	.short	7
	.byte	0
	.byte	8
	.align	3

.subsections_via_symbols
