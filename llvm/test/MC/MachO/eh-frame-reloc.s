// RUN: llvm-mc < %s -triple=x86_64-apple-macosx10.7 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=x86_64-apple-macosx10.6 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=x86_64-apple-macosx10.5 -filetype=obj | llvm-readobj -r | FileCheck --check-prefix=OLD64 %s
// RUN: llvm-mc < %s -triple=i686-apple-macosx10.6 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=i686-apple-macosx10.5 -filetype=obj | llvm-readobj -r | FileCheck --check-prefix=OLD32 %s
// RUN: llvm-mc < %s -triple=i686-apple-macosx10.4 -filetype=obj | llvm-readobj -r | FileCheck --check-prefix=OLD32 %s

	.globl	_bar
	.align	4, 0x90
_bar:
	.cfi_startproc
	.cfi_endproc

// CHECK:      Relocations [
// CHECK-NEXT: ]

// OLD32:      Relocations [
// OLD32-NEXT:   Section __eh_frame {
// OLD32-NEXT:     0x20 0 2 n/a GENERIC_RELOC_LOCAL_SECTDIFF 1 -
// OLD32-NEXT:     0x0 0 2 n/a GENERIC_RELOC_PAIR 1 -
// OLD32-NEXT:   }
// OLD32-NEXT: ]

// OLD64:      Relocations [
// OLD64-NEXT:   Section __eh_frame {
// OLD64-NEXT:     0x20 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _bar.eh
// OLD64-NEXT:     0x20 0 3 1 X86_64_RELOC_UNSIGNED 0 _bar
// OLD64-NEXT:   }
// OLD64-NEXT: ]
