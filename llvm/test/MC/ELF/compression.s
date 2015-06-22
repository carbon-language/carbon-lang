// RUN: llvm-mc -filetype=obj -compress-debug-sections -triple x86_64-pc-linux-gnu < %s -o %t
// RUN: llvm-objdump -s %t | FileCheck %s
// RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck --check-prefix=INFO %s
// RUN: llvm-mc -filetype=obj -compress-debug-sections -triple i386-pc-linux-gnu < %s \
// RUN:     | llvm-readobj -symbols - | FileCheck --check-prefix=386-SYMBOLS %s

// REQUIRES: zlib

// CHECK: Contents of section .zdebug_line:
// Check for the 'ZLIB' file magic at the start of the section only
// CHECK-NEXT: ZLIB
// CHECK-NOT: ZLIB

// Don't compress small sections, such as this simple debug_abbrev example
// CHECK: Contents of section .debug_abbrev:
// CHECK-NOT: ZLIB
// CHECK-NOT: Contents of

// CHECK: Contents of section .debug_info:

// FIXME: Handle compressing alignment fragments to support compressing debug_frame
// CHECK: Contents of section .debug_frame:
// CHECK-NOT: ZLIB
// CHECK: Contents of

// Decompress one valid dwarf section just to check that this roundtrips
// INFO: 0x00000000: Compile Unit: length = 0x0000000c version = 0x0004 abbr_offset = 0x0000 addr_size = 0x08 (next unit at 0x00000010)

// In x86 32 bit named symbols are used for temporary symbols in merge
// sections, so make sure we handle symbols inside compressed sections
// 386-SYMBOLS:        Symbol {
// 386-SYMBOLS-NEXT:    Name:  (0)
// 386-SYMBOLS-NEXT:    Value: 0x0
// 386-SYMBOLS-NEXT:    Size: 0
// 386-SYMBOLS-NEXT:    Binding: Local (0x0)
// 386-SYMBOLS-NEXT:    Type: None (0x0)
// 386-SYMBOLS-NEXT:    Other: 0
// 386-SYMBOLS-NEXT:    Section: Undefined (0x0)
// 386-SYMBOLS-NEXT:  }
// 386-SYMBOLS-NEXT:  Symbol {
// 386-SYMBOLS-NEXT:    Name:  (0)
// 386-SYMBOLS-NEXT:    Value: 0x0
// 386-SYMBOLS-NEXT:    Size: 0
// 386-SYMBOLS-NEXT:    Binding: Local (0x0)
// 386-SYMBOLS-NEXT:    Type: None (0x0)
// 386-SYMBOLS-NEXT:    Other: 0
// 386-SYMBOLS-NEXT:    Section: .zdebug_str (0xA)
// 386-SYMBOLS-NEXT:  }

	.section	.debug_line,"",@progbits

	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.section	.debug_info,"",@progbits
	.long	12                      # Length of Unit
	.short	4                       # DWARF version number
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_comp_dir

	.text
foo:
	.cfi_startproc
	.file 1 "Driver.ii"
# pad out the line table to make sure it's big enough to warrant compression
	.loc 1 2 0
        nop
	.loc 1 3 0
        nop
	.loc 1 4 0
        nop
	.loc 1 5 0
        nop
	.loc 1 6 0
        nop
	.loc 1 7 0
        nop
	.loc 1 8 0
        nop
	.cfi_endproc
	.cfi_sections .debug_frame

	.section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "compress this                                    "
