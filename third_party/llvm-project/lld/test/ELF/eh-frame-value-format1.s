# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
# RUN: ld.lld --eh-frame-hdr %t -o /dev/null

.section .eh_frame,"a",@unwind
  .byte 0x14
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x01
  
  .byte 0x50 # Augmentation string: 'P','\0'
  .byte 0x00
  
  .byte 0x01
  
  .byte 0x01 # LEB128
  .byte 0x01 # LEB128

  .byte 0x04 # DW_EH_PE_udata8
  .byte 0xFF
  .byte 0xFF
  .byte 0xFF
  .byte 0xFF
  .byte 0xFF
  .byte 0xFF
  .byte 0xFF
  .byte 0xFF
  
  .byte 0xFF
