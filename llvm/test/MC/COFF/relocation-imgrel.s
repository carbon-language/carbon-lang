// COFF Image-relative relocations
//
// Test that we produce image-relative relocations (IMAGE_REL_I386_DIR32NB
// and IMAGE_REL_AMD64_ADDR32NB) when accessing foo@imgrel.

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s | llvm-readobj -r | FileCheck --check-prefix=W32 %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s | llvm-readobj -r | FileCheck --check-prefix=W64 %s

.data
foo:
    .long 1

.text
    mov foo@IMGREL(%ebx, %ecx, 4), %eax
    mov foo@imgrel(%ebx, %ecx, 4), %eax

// W32:      Relocations [
// W32-NEXT:   Section (1) .text {
// W32-NEXT:     0x3 IMAGE_REL_I386_DIR32NB foo
// W32-NEXT:     0xA IMAGE_REL_I386_DIR32NB foo
// W32-NEXT:   }
// W32-NEXT: ]

// W64:      Relocations [
// W64-NEXT:   Section (1) .text {
// W64-NEXT:     0x4 IMAGE_REL_AMD64_ADDR32NB foo
// W64-NEXT:     0xC IMAGE_REL_AMD64_ADDR32NB foo
// W64-NEXT:   }
// W64-NEXT: ]
