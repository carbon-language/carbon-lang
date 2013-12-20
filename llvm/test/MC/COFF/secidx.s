// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -s -sr | FileCheck %s

// check that we produce the correct relocation for .secidx

Lfoo:
	.secidx	Lfoo
	.secidx	Lbar

.section spam
Lbar:
	ret

// CHECK:       Relocations [
// CHECK-NEXT:    0x0 IMAGE_REL_I386_SECTION .text
// CHECK-NEXT:    0x4 IMAGE_REL_I386_SECTION spam
// CHECK-NEXT:  ]
