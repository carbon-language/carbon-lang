; RUN: llc -O0 -filetype=asm -mtriple=armv7-linux-gnuehabi < %s | FileCheck %s
;
; Generated with clang with source
; __thread int x;

@x = thread_local global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

; 6 byte of data
; CHECK: .byte 6 @ DW_AT_location
; DW_OP_const4u
; CHECK: .byte 12
; The debug relocation of the address of the tls variable
; CHECK: .long x(tlsldo)

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/tls.c] [DW_LANG_C99]
!1 = metadata !{metadata !"tls.c", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x34\00x\00x\00\001\000\001", null, metadata !5, metadata !6, i32* @x, null} ; [ DW_TAG_variable ] [x] [line 1] [def]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/tls.c]
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!9 = metadata !{metadata !"clang version 3.5 "}
