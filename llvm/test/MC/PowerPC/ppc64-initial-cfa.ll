; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=obj -relocation-model=static %s -o - | \
; RUN: llvm-readobj -s -sr -sd | FileCheck %s -check-prefix=STATIC
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -filetype=obj -relocation-model=pic %s -o - | \
; RUN: llvm-readobj -s -sr -sd | FileCheck %s -check-prefix=PIC

; FIXME: this file should be in .s form, change when asm parser is available.

define void @f() {
entry:
  ret void
}

; STATIC:      Section {
; STATIC:        Name: .eh_frame
; STATIC-NEXT:   Type: SHT_PROGBITS
; STATIC-NEXT:   Flags [ (0x2)
; STATIC-NEXT:     SHF_ALLOC
; STATIC-NEXT:   ]
; STATIC-NEXT:   Address:
; STATIC-NEXT:   Offset:
; STATIC-NEXT:   Size: 40
; STATIC-NEXT:   Link: 0
; STATIC-NEXT:   Info: 0
; STATIC-NEXT:   AddressAlignment: 8
; STATIC-NEXT:   EntrySize: 
; STATIC-NEXT:   Relocations [
; STATIC-NEXT:     0x1C R_PPC64_REL32 .text 0x0
; STATIC-NEXT:   ]
; STATIC-NEXT:   SectionData (
; STATIC-NEXT:     0000: 00000010 00000000 017A5200 01784101
; STATIC-NEXT:     0010: 1B0C0100 00000010 00000018 00000000
; STATIC-NEXT:     0020: 00000010 00000000
; STATIC-NEXT:   )
; STATIC-NEXT: }

; STATIC:      Section {
; STATIC:        Name: .rela.eh_frame
; STATIC-NEXT:   Type: SHT_RELA
; STATIC-NEXT:   Flags [ (0x0)
; STATIC-NEXT:   ]
; STATIC-NEXT:   Address:
; STATIC-NEXT:   Offset:
; STATIC-NEXT:   Size: 24
; STATIC-NEXT:   Link:
; STATIC-NEXT:   Info:
; STATIC-NEXT:   AddressAlignment: 8
; STATIC-NEXT:   EntrySize: 24


; PIC:      Section {
; PIC:        Name: .eh_frame
; PIC-NEXT:   Type: SHT_PROGBITS
; PIC-NEXT:   Flags [ (0x2)
; PIC-NEXT:     SHF_ALLOC
; PIC-NEXT:   ]
; PIC-NEXT:   Address:
; PIC-NEXT:   Offset:
; PIC-NEXT:   Size: 40
; PIC-NEXT:   Link: 0
; PIC-NEXT:   Info: 0
; PIC-NEXT:   AddressAlignment: 8
; PIC-NEXT:   EntrySize: 0
; PIC-NEXT:   Relocations [
; PIC-NEXT:     0x1C R_PPC64_REL32 .text 0x0
; PIC-NEXT:   ]
; PIC-NEXT:   SectionData (
; PIC-NEXT:     0000: 00000010 00000000 017A5200 01784101
; PIC-NEXT:     0010: 1B0C0100 00000010 00000018 00000000
; PIC-NEXT:     0020: 00000010 00000000
; PIC-NEXT:   )
; PIC-NEXT: }

; PIC:      Section {
; PIC:        Name: .rela.eh_frame
; PIC-NEXT:   Type: SHT_RELA
; PIC-NEXT:   Flags [ (0x0)
; PIC-NEXT:   ]
; PIC-NEXT:   Address:
; PIC-NEXT:   Offset:
; PIC-NEXT:   Size: 24
; PIC-NEXT:   Link:
; PIC-NEXT:   Info:
; PIC-NEXT:   AddressAlignment: 8
; PIC-NEXT:   EntrySize: 24
