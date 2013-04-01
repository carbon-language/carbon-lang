// This just tests that a relocation of the specified type shows up as the first
// relocation in the relocation section for .eh_frame when produced by the
// assembler.

// RUN: llvm-mc -filetype=obj %s -o - -triple mips-unknown-unknown | \
// RUN: llvm-objdump -r - | FileCheck --check-prefix=MIPS32 %s

// RUN: llvm-mc -filetype=obj %s -o - -triple mips64-unknown-unknown | \
// RUN: llvm-objdump -r - | FileCheck --check-prefix=MIPS64 %s

// PR15448

func:
	.cfi_startproc
	.cfi_endproc

// MIPS32: RELOCATION RECORDS FOR [.eh_frame]:
// MIPS32-NEXT: R_MIPS_32

// MIPS64: RELOCATION RECORDS FOR [.eh_frame]:
// MIPS64-NEXT: R_MIPS_64
