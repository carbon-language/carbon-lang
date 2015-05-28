// Verify the .fix data section conveys the right offsets and the right relocations
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o - | llvm-readobj -relocations -expand-relocs -sections -section-data | FileCheck %s --check-prefix=READOBJ

        .text
	.section	.rdata,"dr"
	.globl	g1                      # @g1
	.align	4
g1:
	.long	1                       # 0x1

	.globl	g2                      # @g2
	.align	4
g2:
	.long	2                       # 0x2

	.globl	g3                      # @g3
	.align	4
g3:
	.long	3                       # 0x3

	.globl	g4                      # @g4
	.align	4
g4:
	.long	4                       # 0x4

	.section	.fix,"dw"
	.globl	t1                      # @t1
	.align	8
t1:
	.quad	(g3-t1)+4

	.globl	t2                      # @t2
	.align	8
t2:
	.quad	g3-t2

	.globl	t3                      # @t3
	.align	8
t3:
	.quad	(g3-t3)-4

	.globl	t4                      # @t4
	.align	4
t4:
	.long	g3-t4

	.globl	t5                      # @t5
	.align	4
t5:
	.long	g3@IMGREL

	.globl	t6                      # @t6
	.align	16
t6:
	.ascii	"\001\002"
	.zero	6
	.quad	256                     # 0x100
	.long	g3-(t6+16)
	.zero	4


// READOBJ:  Section {
// READOBJ:    Number: 5
// READOBJ:    Name: .fix (2E 66 69 78 00 00 00 00)
// READOBJ:    VirtualSize: 0x0
// READOBJ:    VirtualAddress: 0x0
// READOBJ:    RawDataSize: 56
// READOBJ:    PointerToRawData: 0xEC
// READOBJ:    PointerToRelocations: 0x124
// READOBJ:    PointerToLineNumbers: 0x0
// READOBJ:    RelocationCount: 6
// READOBJ:    LineNumberCount: 0
// READOBJ:    Characteristics [ (0xC0500040)
// READOBJ:      IMAGE_SCN_ALIGN_16BYTES (0x500000)
// READOBJ:      IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
// READOBJ:      IMAGE_SCN_MEM_READ (0x40000000)
// READOBJ:      IMAGE_SCN_MEM_WRITE (0x80000000)
// READOBJ:    ]
// READOBJ:    SectionData (
// READOBJ:      0000: 10000000 00000000 0C000000 00000000  |................|
// READOBJ:      0010: 08000000 00000000 0C000000 00000000  |................|
// READOBJ:      0020: 01020000 00000000 00010000 00000000  |................|
// READOBJ:      0030: 0C000000 00000000                    |........|
// READOBJ:    )
// READOBJ:  }
// READOBJ:  ]
// READOBJ:  Relocations [
// READOBJ:  Section (5) .fix {
// READOBJ:    Relocation {
// READOBJ:      Offset: 0x0
// READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ:      Symbol: .rdata
// READOBJ:    }
// READOBJ:    Relocation {
// READOBJ:      Offset: 0x8
// READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ:      Symbol: .rdata
// READOBJ:    }
// READOBJ:    Relocation {
// READOBJ:      Offset: 0x10
// READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ:      Symbol: .rdata
// READOBJ:    }
// READOBJ:    Relocation {
// READOBJ:      Offset: 0x18
// READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ:      Symbol: .rdata
// READOBJ:    }
// READOBJ:    Relocation {
// READOBJ:      Offset: 0x1C
// READOBJ:      Type: IMAGE_REL_AMD64_ADDR32NB (3)
// READOBJ:      Symbol: g3
// READOBJ:    }
// READOBJ:    Relocation {
// READOBJ:      Offset: 0x30
// READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
// READOBJ:      Symbol: .rdata
// READOBJ:    }
