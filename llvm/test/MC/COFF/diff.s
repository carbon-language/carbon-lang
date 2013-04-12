// RUN: llvm-mc -filetype=obj -triple i686-pc-mingw32 %s | llvm-readobj -s -sr -sd | FileCheck %s

	.def	 _foobar;
	.scl	2;
	.type	32;
	.endef
	.text
	.long   0
	.globl	_foobar
	.align	16, 0x90
_foobar:                                # @foobar
# BB#0:
	ret

	.data
	.globl	_rust_crate             # @rust_crate
	.align	4
_rust_crate:
	.long   0
	.long   _foobar
	.long	_foobar-_rust_crate
	.long	_foobar-_rust_crate

// CHECK:        Name: .data
// CHECK:        Relocations [
// CHECK-NEXT:     0x4 IMAGE_REL_I386_DIR32 _foobar
// CHECK-NEXT:     0x8 IMAGE_REL_I386_REL32 .text
// CHECK-NEXT:     0xC IMAGE_REL_I386_REL32 .text
// CHECK-NEXT:   ]
// CHECK:        SectionData (
// CHECK-NEXT:     0000: 00000000 00000000 1C000000 20000000
// CHECK-NEXT:   )
