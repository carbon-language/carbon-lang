@ RUN: not llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
@ RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s
@ rdar://16335232

.eabi_attribute 8, 1
@ CHECK-ERROR: error: .eabi_attribute directive not valid for Mach-O

.cpu
@ CHECK-ERROR: error: .cpu directive not valid for Mach-O

.fpu neon
@ CHECK-ERROR: error: .fpu directive not valid for Mach-O

.arch armv7
@ CHECK-ERROR: error: .arch directive not valid for Mach-O

.fnstart
@ CHECK-ERROR: error: .fnstart directive not valid for Mach-O

.tlsdescseq
@ CHECK-ERROR: error: .tlsdescseq directive not valid for Mach-O

.object_arch armv7
@ CHECK-ERROR: error: .object_arch directive not valid for Mach-O
