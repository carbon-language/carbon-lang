// Test the bits of .eh_frame on mips that are already implemented correctly.

// FIXME: This test would be a lot cleaner if llvm-objdump had the
// --dwarf=frames option.

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mips
// RUN: llvm-objdump -r -s %t.o | FileCheck --check-prefix=MIPS32 %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mipsel
// RUN: llvm-objdump -r -s %t.o | FileCheck --check-prefix=MIPS32EL %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mips64
// RUN: llvm-objdump -r -s %t.o | FileCheck --check-prefix=MIPS64 %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mips64el
// RUN: llvm-objdump -r -s %t.o | FileCheck --check-prefix=MIPS64EL %s

func:
	.cfi_startproc
	.cfi_endproc

// MIPS32: RELOCATION RECORDS FOR [.eh_frame]:
// MIPS32-NEXT: R_MIPS_32
// MIPS32: Contents of section .eh_frame:
// MIPS32-NEXT: 0000

// Length
// MIPS32: 00000010

// CIE ID
// MIPS32: 00000000

// Version
// MIPS32: 01

// Augmentation String
// MIPS32: 7a5200

// Code Alignment Factor
// MIPS32: 01

// Data Alignment Factor (-4)
// MIPS32: 7c

// Return Address Register
// MIPS32: 1f

// Augmentation Size
// MIPS32: 01

// MIPS32: .........zR..|..
// MIPS32-NEXT: 0010

// Augmentation (fde pointer encoding: DW_EH_PE_sdata4)
// MIPS32: 0b
// FIXME: The instructions are different from the ones produces by gas.

// MIPS32EL: RELOCATION RECORDS FOR [.eh_frame]:
// MIPS32EL-NEXT: R_MIPS_32
// MIPS32EL: Contents of section .eh_frame:
// MIPS32EL-NEXT: 0000

// Length
// MIPS32EL: 10000000

// CIE ID
// MIPS32EL: 00000000

// Version
// MIPS32EL: 01

// Augmentation String
// MIPS32EL: 7a5200

// Code Alignment Factor
// MIPS32EL: 01

// Data Alignment Factor (-4)
// MIPS32EL: 7c

// Return Address Register
// MIPS32EL: 1f

// Augmentation Size
// MIPS32EL: 01

// MIPS32EL: .........zR..|..
// MIPS32EL-NEXT: 0010

// Augmentation (fde pointer encoding: DW_EH_PE_sdata4)
// MIPS32EL: 0b
// FIXME: The instructions are different from the ones produces by gas.

// MIPS64: RELOCATION RECORDS FOR [.eh_frame]:
// MIPS64-NEXT: R_MIPS_64
// MIPS64: Contents of section .eh_frame:
// MIPS64-NEXT: 0000

// Length
// MIPS64: 00000010

// CIE ID
// MIPS64: 00000000

// Version
// MIPS64: 01

// Augmentation String
// MIPS64: 7a5200

// Code Alignment Factor
// MIPS64: 01

// Data Alignment Factor (-8). GAS uses -4. Should be ok as long as all
// offsets we need are a multiple of 8.
// MIPS64: 78

// Return Address Register
// MIPS64: 1f

// Augmentation Size
// MIPS64: 01

// MIPS64: .........zR..x..
// MIPS64-NEXT: 0010

// Augmentation (fde pointer encoding: DW_EH_PE_sdata8)
// MIPS64: 0c
// FIXME: The instructions are different from the ones produces by gas.


// MIPS64EL: RELOCATION RECORDS FOR [.eh_frame]:
// FIXME: llvm-objdump currently misprints the relocations for mips64el
// MIPS64EL: Contents of section .eh_frame:
// MIPS64EL-NEXT: 0000

// Length
// MIPS64EL: 10000000

// CIE ID
// MIPS64EL: 00000000

// Version
// MIPS64EL: 01

// Augmentation String
// MIPS64EL: 7a5200

// Code Alignment Factor
// MIPS64EL: 01

// Data Alignment Factor (-8). GAS uses -4. Should be ok as long as all
// offsets we need are a multiple of 8.
// MIPS64EL: 78

// Return Address Register
// MIPS64EL: 1f

// Augmentation Size
// MIPS64EL: 01

// MIPS64EL: .........zR..x..
// MIPS64EL-NEXT: 0010

// Augmentation (fde pointer encoding: DW_EH_PE_sdata8)
// MIPS64EL: 0c
// FIXME: The instructions are different from the ones produces by gas.
