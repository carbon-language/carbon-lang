; RUN: llc %s -mtriple=thumbv5-linux-gnueabi -mcpu=xscale -o - | \
; RUN: FileCheck -check-prefix=ASM %s

; RUN: llc %s -mtriple=thumbv5-linux-gnueabi -filetype=obj \
; RUN: -mcpu=xscale -o - | llvm-readobj -s -sd | \
; RUN: FileCheck -check-prefix=OBJ %s

; FIXME: The OBJ test should be a .s to .o test and the ASM test should
; be moved to test/CodeGen/ARM.

define void @foo() nounwind {
entry:
  ret void
}

; ASM:           .eabi_attribute 6, 5
; ASM-NEXT:      .eabi_attribute 8, 1
; ASM-NEXT:      .eabi_attribute 9, 1

; OBJ:      Sections [
; OBJ:        Section {
; OBJ:          Index: 4
; OBJ-NEXT:     Name: .ARM.attributes (12)
; OBJ-NEXT:     Type: SHT_ARM_ATTRIBUTES
; OBJ-NEXT:     Flags [ (0x0)
; OBJ-NEXT:     ]
; OBJ-NEXT:     Address: 0x0
; OBJ-NEXT:     Offset: 0x38
; OBJ-NEXT:     Size: 40
; OBJ-NEXT:     Link: 0
; OBJ-NEXT:     Info: 0
; OBJ-NEXT:     AddressAlignment: 1
; OBJ-NEXT:     EntrySize: 0
; OBJ-NEXT:     SectionData (
; OBJ-NEXT:       0000: 41270000 00616561 62690001 1D000000
; OBJ-NEXT:       0010: 05585343 414C4500 06050801 09011401
; OBJ-NEXT:       0020: 15011703 18011901
; OBJ-NEXT:     )
; OBJ-NEXT:   }
