@ Check multiple .fpu directives.

@ The later .fpu directive should overwrite the earlier one.
@ See also: directive-fpu-multiple2.s.

@ RUN: llvm-mc -triple arm-eabi -filetype obj %s | llvm-readobj -arm-attributes \
@ RUN:   | FileCheck %s -check-prefix CHECK-ATTR

	.fpu neon
	.fpu vfpv4

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: FP_arch
@ CHECK-ATTR:     Description: VFPv4
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

