; The purpose of this test is to verify that weak linkage type is not ignored by backend,
; if section was specialized.

; RUN: llc -filetype=obj -mtriple i686-pc-win32 %s -o - | coff-dump.py | FileCheck %s

@a = weak unnamed_addr constant { i32, i32, i32 } { i32 0, i32 0, i32 0}, section ".data"

; CHECK:           Name                     = .data$a
; CHECK-NEXT:      VirtualSize              = 0
; CHECK-NEXT:      VirtualAddress           = 0
; CHECK-NEXT:      SizeOfRawData            = {{[0-9]+}}
; CHECK-NEXT:      PointerToRawData         = 0x{{[0-9A-F]+}}
; CHECK-NEXT:      PointerToRelocations     = 0x0
; CHECK-NEXT:      PointerToLineNumbers     = 0x0
; CHECK-NEXT:      NumberOfRelocations      = 0
; CHECK-NEXT:      NumberOfLineNumbers      = 0
; CHECK-NEXT:      Charateristics           = 0x40401040
; CHECK-NEXT:        IMAGE_SCN_CNT_INITIALIZED_DATA
; CHECK-NEXT:        IMAGE_SCN_LNK_COMDAT
; CHECK-NEXT:        IMAGE_SCN_ALIGN_8BYTES
; CHECK-NEXT:        IMAGE_SCN_MEM_READ
; CHECK-NEXT:      SectionData              = 
; CHECK-NEXT:        00 00 00 00 00 00 00 00 - 00 00 00 00 
