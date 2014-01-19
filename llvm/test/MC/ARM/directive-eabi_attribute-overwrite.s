@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -s -sd \
@ RUN:   | FileCheck %s

	.syntax unified
	.thumb

	.eabi_attribute Tag_compatibility, 1
	.eabi_attribute Tag_compatibility, 1, "aeabi"

@ CHECK: Section {
@ CHECK:   Name: .ARM.attributes
@ CHECK:   Type: SHT_ARM_ATTRIBUTES
@ CHECK:   SectionData (
@ CHECK:     0000: 41170000 00616561 62690001 0D000000
@ CHECK:     0010: 20014145 41424900
@ CHECK:   )
@ CHECK: }

