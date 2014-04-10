// RUN: llvm-mc -filetype=obj -compress-debug-sections -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-objdump -s %t | FileCheck %s
// RUN: llvm-dwarfdump -debug-dump=abbrev %t | FileCheck --check-prefix=ABBREV %s

// REQUIRES: zlib

// CHECK: Contents of section .zdebug_line:
// Check for the 'ZLIB' file magic at the start of the section only
// CHECK-NEXT: ZLIB
// CHECK-NOT: ZLIB
// CHECK: Contents of

// CHECK: Contents of section .zdebug_abbrev:
// CHECK-NEXT: ZLIB

// FIXME: Handle compressing alignment fragments to support compressing debug_frame
// CHECK: Contents of section .debug_frame:
// CHECK-NOT: ZLIB
// CHECK: Contents of

// Decompress one valid dwarf section just to check that this roundtrips
// ABBREV: Abbrev table for offset: 0x00000000
// ABBREV: [1] DW_TAG_compile_unit DW_CHILDREN_no

	.section	.debug_line,"",@progbits

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.text
foo:
	.cfi_startproc
	.file 1 "Driver.ii"
	.loc 1 2 0
        nop
	.cfi_endproc
	.cfi_sections .debug_frame
