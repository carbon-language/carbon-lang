// RUN: llvm-mc -filetype=obj -compress-debug-sections -triple x86_64-pc-linux-gnu %s -o - | llvm-objdump -s - | FileCheck %s

// REQUIRES: zlib

// CHECK: Contents of section .debug_line:
// FIXME: Figure out how to handle debug_line that currently uses multiple section fragments
// CHECK-NOT: ZLIB

// CHECK: Contents of section .zdebug_abbrev:
// Check for the 'ZLIB' file magic at the start of the section
// CHECK-NEXT: ZLIB

// We shouldn't compress the debug_frame section, since it can be relaxed
// CHECK: Contents of section .debug_frame
// CHECK-NOT: ZLIB

	.section	.debug_line,"",@progbits

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.text
foo:
	.cfi_startproc
	.file 1 "Driver.ii"
	.loc 1 2 0
        nop
	.cfi_endproc
	.cfi_sections .debug_frame
