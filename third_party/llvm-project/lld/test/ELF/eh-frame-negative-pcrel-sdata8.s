# REQUIRES: x86

# Test handling of FDE pc negative relative addressing with DW_EH_PE_sdata8.
# This situation can arise when .eh_frame is placed after .text.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { .text : { *(.text) } .eh_frame : { *(.eh_frame) } }" > %t.script
# RUN: ld.lld --eh-frame-hdr --script %t.script --section-start .text=0x1000 %t.o -o %t
# RUN: llvm-readobj -S --section-data %t | FileCheck %s

# CHECK:      Section {
# CHECK:        Index:
# CHECK:        Name: .eh_frame
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x1001
# CHECK-NEXT:   Offset: 0x1001
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT:   SectionData (
# CHECK-NEXT:     0000: 14000000 00000000 017A5200 01010101
# CHECK-NEXT:     0010: 1C000000 00000000 14000000 1C000000
# CHECK-NEXT:     0020: DFFFFFFF FFFFFFFF
#                       ^
#   DFFFFFFF FFFFFFFF = _start(0x1000) - PC(.eh_frame(0x1001) + 0x20)

# CHECK:      Section {
# CHECK:        Index:
# CHECK:        Name: .eh_frame_hdr
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x1038
# CHECK-NEXT:   Offset: 0x1038
# CHECK-NEXT:   Size: 20
# CHECK-NEXT:   Link: 0
# CHECK-NEXT:   Info: 0
# CHECK-NEXT:   AddressAlignment: 4
# CHECK-NEXT:   EntrySize: 0
# CHECK-NEXT:   SectionData (
# CHECK-NEXT:     0000: 011B033B C5FFFFFF 01000000 C8FFFFFF
# CHECK-NEXT:     0010: E1FFFFFF
#   Header (always 4 bytes): 011B033B
#   C5FFFFFF = .eh_frame(0x1001) - .eh_frame_hdr(0x1038) - 4
#   01000000 = 1 = the number of FDE pointers in the table.
#   C8FFFFFF = _start(0x1000) - .eh_frame_hdr(0x1038)
#   E1FFFFFF = FDE(.eh_frame(0x1001) + 0x18) - .eh_frame_hdr(0x1038)

.text
.global _start
_start:
 nop

.section .eh_frame,"a",@unwind
  .long 16   # Size
  .long 0x00 # ID
  .byte 0x01 # Version.

  .byte 0x7A # Augmentation string: "zR"
  .byte 0x52
  .byte 0x00

  .byte 0x01

  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x01 # LEB128
  .byte 0x1C # DW_EH_PE_pcrel | DW_EH_PE_sdata8

  .byte 0x00
  .byte 0x00
  .byte 0x00

  .long 16   # Size
  .long 24   # ID
fde:
  .quad _start - fde
  .long 0
