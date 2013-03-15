// RUN: llvm-mc -filetype=obj %s -o - -triple x86_64-pc-linux | llvm-objdump -r - | FileCheck --check-prefix=X86-64 %s
// RUN: llvm-mc -filetype=obj %s -o - -triple i686-pc-linux | llvm-objdump -r - | FileCheck --check-prefix=I686 %s

// PR15448

func:
	.cfi_startproc
	.cfi_endproc

// X86-64: R_X86_64_PC32
// I686: R_386_PC32
