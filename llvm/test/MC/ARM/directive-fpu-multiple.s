@ Check multiple .fpu directives.

@ The later .fpu directive should overwrite the earlier one.
@ See also: directive-fpu-multiple2.s.

@ RUN: llvm-mc < %s -triple arm-unknown-linux-gnueabi -filetype=obj \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s

	.fpu neon
	.fpu vfpv4

@ CHECK:      Name: .ARM.attributes
@ CHECK-NEXT: Type: SHT_ARM_ATTRIBUTES (0x70000003)
@ CHECK-NEXT: Flags [ (0x0)
@ CHECK-NEXT: ]
@ CHECK-NEXT: Address: 0x0
@ CHECK-NEXT: Offset: 0x34
@ CHECK-NEXT: Size: 18
@ CHECK-NEXT: Link: 0
@ CHECK-NEXT: Info: 0
@ CHECK-NEXT: AddressAlignment: 1
@ CHECK-NEXT: EntrySize: 0
@ CHECK-NEXT: SectionData (
@ CHECK-NEXT:   0000: 41110000 00616561 62690001 07000000
@ CHECK-NEXT:   0010: 0A05
@ CHECK-NEXT: )
