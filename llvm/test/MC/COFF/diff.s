// RUN: llvm-mc -filetype=obj -triple i686-pc-mingw32 %s | coff-dump.py | FileCheck %s

	.def	 _foobar;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_foobar
	.align	16, 0x90
_foobar:                                # @foobar
# BB#0:
	ret

	.data
	.globl	_rust_crate             # @rust_crate
	.align	4
_rust_crate:
	.long	_foobar-_rust_crate


// CHECK:       Relocations              = [
// CHECK-NEXT:   0 = {
// CHECK-NEXT:     VirtualAddress           = 0x0
// CHECK-NEXT:     SymbolTableIndex         =
// CHECK-NEXT:     Type                     = IMAGE_REL_I386_REL32 (20)
// CHECK-NEXT:     SymbolName               = .text
// CHECK-NEXT:   }
// CHECK-NEXT: ]
