; Test the sh_link in Elf32_Shdr.

; The .ARM.exidx section should be linked with corresponding text section.
; The sh_link in Elf32_Shdr should be filled with the section index of
; the text section.

; RUN: llc -mtriple arm-unknown-linux-gnueabi \
; RUN:     -arm-enable-ehabi -arm-enable-ehabi-descriptors \
; RUN:     -filetype=obj -o - %s \
; RUN:   | llvm-readobj -s \
; RUN:   | FileCheck %s

define void @test1() nounwind {
entry:
  ret void
}

define void @test2() nounwind section ".test_section" {
entry:
  ret void
}

; CHECK:      Sections [
; CHECK:        Section {
; CHECK:          Index: 1
; CHECK-NEXT:     Name: .text (16)

; CHECK:        Section {
; CHECK:          Name: .ARM.exidx (5)
; CHECK-NEXT:     Type: SHT_ARM_EXIDX
; CHECK-NEXT:     Flags [ (0x82)
; CHECK-NEXT:       SHF_ALLOC
; CHECK-NEXT:       SHF_LINK_ORDER
; CHECK-NEXT:     ]
; CHECK-NEXT:     Address: 0x0
; CHECK-NEXT:     Offset: 0x5C
; CHECK-NEXT:     Size: 8
; CHECK-NEXT:     Link: 1
; CHECK-NEXT:     Info: 0
; CHECK-NEXT:     AddressAlignment: 4

; CHECK:        Section {
; CHECK:          Index: 7
; CHECK-NEXT:     Name: .test_section (57)

; CHECK:        Section {
; CHECK:          Name: .ARM.exidx.test_section (47)
; CHECK-NEXT:     Type: SHT_ARM_EXIDX
; CHECK-NEXT:     Flags [ (0x82)
; CHECK-NEXT:       SHF_ALLOC
; CHECK-NEXT:       SHF_LINK_ORDER
; CHECK-NEXT:     ]
; CHECK-NEXT:     Address: 0x0
; CHECK-NEXT:     Offset: 0x68
; CHECK-NEXT:     Size: 8
; CHECK-NEXT:     Link: 7
; CHECK-NEXT:     Info: 0
; CHECK-NEXT:     AddressAlignment: 4
