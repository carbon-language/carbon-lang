; Verify the assembler produces the expected expressions
; RUN: llc -mtriple=x86_64-pc-win32 %s -o - | FileCheck %s
; Verify the .fix data section conveys the right offsets and the right relocations
; RUN: llc -mtriple=x86_64-pc-win32 -filetype=obj %s -o - | llvm-readobj -relocations -expand-relocs -sections -section-data | FileCheck %s --check-prefix=READOBJ

;;;; some globals

@g1 = constant i32 1;
@g2 = constant i32 2;
@g3 = constant i32 3;
@g4 = constant i32 4;
@__ImageBase = external global i64*;

;;;; cross-section relative relocations

; CHECK: .quad (g3-t1)+4
@t1 = global i64 add(i64 sub(i64 ptrtoint(i32* @g3 to i64), i64 ptrtoint(i64* @t1 to i64)), i64 4), section ".fix"
; CHECK: .quad g3-t2
@t2 = global i64 sub(i64 ptrtoint(i32* @g3 to i64), i64 ptrtoint(i64* @t2 to i64)), section ".fix"
; CHECK: .quad (g3-t3)-4
@t3 = global i64 sub(i64 sub(i64 ptrtoint(i32* @g3 to i64), i64 ptrtoint(i64* @t3 to i64)), i64 4), section ".fix"
; CHECK: .long g3-t4
@t4 = global i32 trunc(i64 sub(i64 ptrtoint(i32* @g3 to i64), i64 ptrtoint(i32* @t4 to i64)) to i32), section ".fix"

;;;; image base relocation

; CHECK: .long g3@IMGREL
@t5 = global i32 trunc(i64 sub(i64 ptrtoint(i32* @g3 to i64), i64 ptrtoint(i64** @__ImageBase to i64)) to i32), section ".fix"

;;;; cross-section relative with source offset

%struct.EEType = type { [2 x i8], i64, i32}

; CHECK: .long g3-(t6+16)
@t6 = global %struct.EEType { 
        [2 x i8] c"\01\02", 
        i64 256,
        i32 trunc(i64 sub(i64 ptrtoint(i32* @g3 to i64), i64 ptrtoint(i32* getelementptr inbounds (%struct.EEType, %struct.EEType* @t6, i32 0, i32 2) to i64)) to i32 )
}, section ".fix"

; READOBJ:  Section {
; READOBJ:    Number: 5
; READOBJ:    Name: .fix (2E 66 69 78 00 00 00 00)
; READOBJ:    VirtualSize: 0x0
; READOBJ:    VirtualAddress: 0x0
; READOBJ:    RawDataSize: 56
; READOBJ:    PointerToRawData: 0xEC
; READOBJ:    PointerToRelocations: 0x124
; READOBJ:    PointerToLineNumbers: 0x0
; READOBJ:    RelocationCount: 6
; READOBJ:    LineNumberCount: 0
; READOBJ:    Characteristics [ (0xC0500040)
; READOBJ:      IMAGE_SCN_ALIGN_16BYTES (0x500000)
; READOBJ:      IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
; READOBJ:      IMAGE_SCN_MEM_READ (0x40000000)
; READOBJ:      IMAGE_SCN_MEM_WRITE (0x80000000)
; READOBJ:    ]
; READOBJ:    SectionData (
; READOBJ:      0000: 10000000 00000000 0C000000 00000000  |................|
; READOBJ:      0010: 08000000 00000000 0C000000 00000000  |................|
; READOBJ:      0020: 01020000 00000000 00010000 00000000  |................|
; READOBJ:      0030: 0C000000 00000000                    |........|
; READOBJ:    )
; READOBJ:  }
; READOBJ:  ]
; READOBJ:  Relocations [
; READOBJ:  Section (5) .fix {
; READOBJ:    Relocation {
; READOBJ:      Offset: 0x0
; READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
; READOBJ:      Symbol: .rdata
; READOBJ:    }
; READOBJ:    Relocation {
; READOBJ:      Offset: 0x8
; READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
; READOBJ:      Symbol: .rdata
; READOBJ:    }
; READOBJ:    Relocation {
; READOBJ:      Offset: 0x10
; READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
; READOBJ:      Symbol: .rdata
; READOBJ:    }
; READOBJ:    Relocation {
; READOBJ:      Offset: 0x18
; READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
; READOBJ:      Symbol: .rdata
; READOBJ:    }
; READOBJ:    Relocation {
; READOBJ:      Offset: 0x1C
; READOBJ:      Type: IMAGE_REL_AMD64_ADDR32NB (3)
; READOBJ:      Symbol: g3
; READOBJ:    }
; READOBJ:    Relocation {
; READOBJ:      Offset: 0x30
; READOBJ:      Type: IMAGE_REL_AMD64_REL32 (4)
; READOBJ:      Symbol: .rdata
; READOBJ:    }
