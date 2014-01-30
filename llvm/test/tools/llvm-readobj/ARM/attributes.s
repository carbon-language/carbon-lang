@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj -arm-attributes - | FileCheck %s

	.syntax unified

	.cpu cortex-a8
	.fpu neon

	.eabi_attribute Tag_CPU_raw_name, "Cortex-A9"
	.eabi_attribute Tag_CPU_name, "cortex-a9"
	.eabi_attribute Tag_CPU_arch, 10
	.eabi_attribute Tag_CPU_arch_profile, 'A'
	.eabi_attribute Tag_ARM_ISA_use, 0
	.eabi_attribute Tag_THUMB_ISA_use, 2
	.eabi_attribute Tag_FP_arch, 3
	.eabi_attribute Tag_WMMX_arch, 0
	.eabi_attribute Tag_Advanced_SIMD_arch, 1
	.eabi_attribute Tag_PCS_config, 2
	.eabi_attribute Tag_ABI_PCS_R9_use, 0
	.eabi_attribute Tag_ABI_PCS_RW_data, 0
	.eabi_attribute Tag_ABI_PCS_RO_data, 0
	.eabi_attribute Tag_ABI_PCS_GOT_use, 0
	.eabi_attribute Tag_ABI_PCS_wchar_t, 4
	.eabi_attribute Tag_ABI_FP_rounding, 1
	.eabi_attribute Tag_ABI_FP_denormal, 2
	.eabi_attribute Tag_ABI_FP_exceptions, 1
	.eabi_attribute Tag_ABI_FP_user_exceptions, 1
	.eabi_attribute Tag_ABI_FP_number_model, 3
	.eabi_attribute Tag_ABI_align_needed, 1
	.eabi_attribute Tag_ABI_align_preserved, 2
	.eabi_attribute Tag_ABI_enum_size, 3
	.eabi_attribute Tag_ABI_HardFP_use, 0
	.eabi_attribute Tag_ABI_VFP_args, 1
	.eabi_attribute Tag_ABI_WMMX_args, 0
	.eabi_attribute Tag_ABI_optimization_goals, 2
	.eabi_attribute Tag_ABI_FP_optimization_goals, 2
	.eabi_attribute Tag_compatibility, 1
	.eabi_attribute Tag_compatibility, 1, "aeabi"
	.eabi_attribute Tag_CPU_unaligned_access, 0
	.eabi_attribute Tag_FP_HP_extension, 0
	.eabi_attribute Tag_ABI_FP_16bit_format, 0
	.eabi_attribute Tag_MPextension_use, 0
	.eabi_attribute Tag_DIV_use, 0
	.eabi_attribute Tag_nodefaults, 0
	.eabi_attribute Tag_also_compatible_with, "gnu"
	.eabi_attribute Tag_T2EE_use, 0
	.eabi_attribute Tag_conformance, "2.09"
	.eabi_attribute Tag_Virtualization_use, 0

@ CHECK: BuildAttributes {
@ CHECK:   Section 1 {
@ CHECK:     Tag: Tag_File (0x1)
@ CHECK:     FileAttributes {
@ CHECK:       Attribute {
@ CHECK:         Tag: 4
@ CHECK:         TagName: CPU_raw_name
@ CHECK:         Value: CORTEX-A9
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 5
@ CHECK:         TagName: CPU_name
@ CHECK:         Value: CORTEX-A9
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 6
@ CHECK:         Value: 10
@ CHECK:         TagName: CPU_arch
@ CHECK:         Description: ARM v7
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 7
@ CHECK:         Value: 65
@ CHECK:         TagName: CPU_arch_profile
@ CHECK:         Description: Application
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 8
@ CHECK:         Value: 0
@ CHECK:         TagName: ARM_ISA_use
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 9
@ CHECK:         Value: 2
@ CHECK:         TagName: THUMB_ISA_use
@ CHECK:         Description: Thumb-2
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 10
@ CHECK:         Value: 3
@ CHECK:         TagName: FP_arch
@ CHECK:         Description: VFPv3
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 11
@ CHECK:         Value: 0
@ CHECK:         TagName: WMMX_arch
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 12
@ CHECK:         Value: 1
@ CHECK:         TagName: Advanced_SIMD_arch
@ CHECK:         Description: NEONv1
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 13
@ CHECK:         Value: 2
@ CHECK:         TagName: PCS_config
@ CHECK:         Description: Linux Application
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 14
@ CHECK:         Value: 0
@ CHECK:         TagName: ABI_PCS_R9_use
@ CHECK:         Description: v6
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 15
@ CHECK:         Value: 0
@ CHECK:         TagName: ABI_PCS_RW_data
@ CHECK:         Description: Absolute
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 16
@ CHECK:         Value: 0
@ CHECK:         TagName: ABI_PCS_RO_data
@ CHECK:         Description: Absolute
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 17
@ CHECK:         Value: 0
@ CHECK:         TagName: ABI_PCS_GOT_use
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 18
@ CHECK:         Value: 4
@ CHECK:         TagName: ABI_PCS_wchar_t
@ CHECK:         Description: 4-byte
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 19
@ CHECK:         Value: 1
@ CHECK:         TagName: ABI_FP_rounding
@ CHECK:         Description: Runtime
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 20
@ CHECK:         Value: 2
@ CHECK:         TagName: ABI_FP_denormal
@ CHECK:         Description: Sign Only
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 21
@ CHECK:         Value: 1
@ CHECK:         TagName: ABI_FP_exceptions
@ CHECK:         Description: IEEE-754
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 22
@ CHECK:         Value: 1
@ CHECK:         TagName: ABI_FP_user_exceptions
@ CHECK:         Description: IEEE-754
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 23
@ CHECK:         Value: 3
@ CHECK:         TagName: ABI_FP_number_model
@ CHECK:         Description: IEEE-754
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 24
@ CHECK:         Value: 1
@ CHECK:         TagName: ABI_align_needed
@ CHECK:         Description: 8-byte alignment
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 25
@ CHECK:         Value: 2
@ CHECK:         TagName: ABI_align_preserved
@ CHECK:         Description: 8-byte data and code alignment
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 26
@ CHECK:         Value: 3
@ CHECK:         TagName: ABI_enum_size
@ CHECK:         Description: External Int32
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 27
@ CHECK:         Value: 0
@ CHECK:         TagName: ABI_HardFP_use
@ CHECK:         Description: Tag_FP_arch
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 28
@ CHECK:         Value: 1
@ CHECK:         TagName: ABI_VFP_args
@ CHECK:         Description: AAPCS VFP
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 29
@ CHECK:         Value: 0
@ CHECK:         TagName: ABI_WMMX_args
@ CHECK:         Description: AAPCS
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 30
@ CHECK:         Value: 2
@ CHECK:         TagName: ABI_optimization_goals
@ CHECK:         Description: Aggressive Speed
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 31
@ CHECK:         Value: 2
@ CHECK:         TagName: ABI_FP_optimization_goals
@ CHECK:         Description: Aggressive Speed
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 32
@ CHECK:         Value: 1, AEABI
@ CHECK:         TagName: compatibility
@ CHECK:         Description: AEABI Conformant
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 34
@ CHECK:         Value: 0
@ CHECK:         TagName: CPU_unaligned_access
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 36
@ CHECK:         Value: 0
@ CHECK:         TagName: FP_HP_extension
@ CHECK:         Description: If Available
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 38
@ CHECK:         Value: 0
@ CHECK:         TagName: ABI_FP_16bit_format
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 42
@ CHECK:         Value: 0
@ CHECK:         TagName: MPextension_use
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 44
@ CHECK:         Value: 0
@ CHECK:         TagName: DIV_use
@ CHECK:         Description: If Available
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 64
@ CHECK:         Value: 0
@ CHECK:         TagName: nodefaults
@ CHECK:         Description: Unspecified Tags UNDEFINED
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 65
@ CHECK:         TagName: also_compatible_with
@ CHECK:         Value: GNU
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 66
@ CHECK:         Value: 0
@ CHECK:         TagName: T2EE_use
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 67
@ CHECK:         TagName: conformance
@ CHECK:         Value: 2.09
@ CHECK:       }
@ CHECK:       Attribute {
@ CHECK:         Tag: 68
@ CHECK:         Value: 0
@ CHECK:         TagName: Virtualization_use
@ CHECK:         Description: Not Permitted
@ CHECK:       }
@ CHECK:     }
@ CHECK:   }
@ CHECK: }

