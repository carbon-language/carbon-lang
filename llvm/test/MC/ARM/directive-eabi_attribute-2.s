@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s

	.syntax unified
	.thumb

	.eabi_attribute Tag_CPU_raw_name, "Cortex-A9"
@ CHECK: .eabi_attribute 4, "Cortex-A9"
	.eabi_attribute Tag_CPU_name, "cortex-a9"
@ CHECK: .cpu cortex-a9
	.eabi_attribute Tag_CPU_arch, 10
@ CHECK: .eabi_attribute 6, 10
	.eabi_attribute Tag_CPU_arch_profile, 'A'
@ CHECK: .eabi_attribute 7, 65
	.eabi_attribute Tag_ARM_ISA_use, 0
@ CHECK: .eabi_attribute 8, 0
	.eabi_attribute Tag_THUMB_ISA_use, 2
@ CHECK: .eabi_attribute 9, 2
	.eabi_attribute Tag_FP_arch, 3
@ CHECK: .eabi_attribute 10, 3
	.eabi_attribute Tag_WMMX_arch, 0
@ CHECK: .eabi_attribute 11, 0
	.eabi_attribute Tag_Advanced_SIMD_arch, 1
@ CHECK: .eabi_attribute 12, 1
	.eabi_attribute Tag_PCS_config, 2
@ CHECK: .eabi_attribute 13, 2
	.eabi_attribute Tag_ABI_PCS_R9_use, 0
@ CHECK: .eabi_attribute 14, 0
	.eabi_attribute Tag_ABI_PCS_RW_data, 0
@ CHECK: .eabi_attribute 15, 0
	.eabi_attribute Tag_ABI_PCS_RO_data, 0
@ CHECK: .eabi_attribute 16, 0
	.eabi_attribute Tag_ABI_PCS_GOT_use, 0
@ CHECK: .eabi_attribute 17, 0
	.eabi_attribute Tag_ABI_PCS_wchar_t, 4
@ CHECK: .eabi_attribute 18, 4
	.eabi_attribute Tag_ABI_FP_rounding, 1
@ CHECK: .eabi_attribute 19, 1
	.eabi_attribute Tag_ABI_FP_denormal, 2
@ CHECK: .eabi_attribute 20, 2
	.eabi_attribute Tag_ABI_FP_exceptions, 1
@ CHECK: .eabi_attribute 21, 1
	.eabi_attribute Tag_ABI_FP_user_exceptions, 1
@ CHECK: .eabi_attribute 22, 1
	.eabi_attribute Tag_ABI_FP_number_model, 3
@ CHECK: .eabi_attribute 23, 3
	.eabi_attribute Tag_ABI_align8_needed, 1
@ CHECK: .eabi_attribute 24, 1
	.eabi_attribute Tag_ABI_align8_preserved, 2
@ CHECK: .eabi_attribute 25, 2
	.eabi_attribute Tag_ABI_enum_size, 3
@ CHECK: .eabi_attribute 26, 3
	.eabi_attribute Tag_ABI_HardFP_use, 0
@ CHECK: .eabi_attribute 27, 0
	.eabi_attribute Tag_ABI_VFP_args, 1
@ CHECK: .eabi_attribute 28, 1
	.eabi_attribute Tag_ABI_WMMX_args, 0
@ CHECK: .eabi_attribute 29, 0
	.eabi_attribute Tag_ABI_FP_optimization_goals, 1
@ CHECK: .eabi_attribute 31, 1
	.eabi_attribute Tag_compatibility, 1
@ CHECK: .eabi_attribute 32, 1
	.eabi_attribute Tag_compatibility, 1, "aeabi"
@ CHECK: .eabi_attribute 32, 1, "aeabi"
	.eabi_attribute Tag_CPU_unaligned_access, 0
@ CHECK: .eabi_attribute 34, 0
	.eabi_attribute Tag_FP_HP_extension, 0
@ CHECK: .eabi_attribute 36, 0
	.eabi_attribute Tag_ABI_FP_16bit_format, 0
@ CHECK: .eabi_attribute 38, 0
	.eabi_attribute Tag_MPextension_use, 0
@ CHECK: .eabi_attribute 42, 0
	.eabi_attribute Tag_DIV_use, 0
@ CHECK: .eabi_attribute 44, 0
	.eabi_attribute Tag_nodefaults, 0
@ CHECK: .eabi_attribute 64, 0
	.eabi_attribute Tag_also_compatible_with, "gnu"
@ CHECK: .eabi_attribute 65, "gnu"
	.eabi_attribute Tag_T2EE_use, 0
@ CHECK: .eabi_attribute 66, 0
	.eabi_attribute Tag_conformance, "2.09"
@ CHECK: .eabi_attribute 67, "2.09"
	.eabi_attribute Tag_Virtualization_use, 0
@ CHECK: .eabi_attribute 68, 0

@ ===--- GNU AS Compatibility Checks ---===

	.eabi_attribute 2 * 2 + 1, "cortex-a9"
@ CHECK: .cpu cortex-a9
	.eabi_attribute 2 * 2 + 2, 5 * 2
@ CHECK: .eabi_attribute 6, 10

