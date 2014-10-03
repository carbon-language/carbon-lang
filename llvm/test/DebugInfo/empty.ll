; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s
; RUN: %llc_dwarf -split-dwarf=Enable < %s -filetype=obj | llvm-dwarfdump - | FileCheck --check-prefix=FISSION %s

; darwin has a workaround for a linker bug so it always emits one line table entry
; XFAIL: darwin

; Expect no line table entry since there are no functions and file references in this compile unit
; CHECK: .debug_line contents:
; CHECK: Line table prologue:
; CHECK: total_length: 0x00000019
; CHECK-NOT: file_names[

; CHECK: .debug_pubnames contents:
; CHECK-NOT: Offset

; CHECK: .debug_pubtypes contents:
; CHECK-NOT: Offset

; Don't emit DW_AT_addr_base when there are no addresses.
; FISSION-NOT: DW_AT_GNU_addr_base [DW_FORM_sec_offset]

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.1 (trunk 143523)\001\00\000\00\000", metadata !4, metadata !2, metadata !2, metadata !2, metadata !2, null} ; [ DW_TAG_compile_unit ]
!2 = metadata !{}
!3 = metadata !{metadata !"0x29", metadata !4} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !"empty.c", metadata !"/home/nlewycky"}
!5 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
