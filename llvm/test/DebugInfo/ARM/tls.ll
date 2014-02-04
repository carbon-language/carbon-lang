; RUN: llc -O0 -filetype=asm -mtriple=armv7-linux-gnuehabi < %s | FileCheck %s
;
; Generated with clang with source
; __thread int x;

@x = thread_local global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

; 6 byte of data
; CHECK: .byte 6 @ DW_AT_location
; DW_OP_const4u
; CHECK: .byte 12
; The debug relocation of the address of the tls variable
; CHECK: .long x(tlsldo)

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/arm.c] [DW_LANG_C99]
!1 = metadata !{metadata !"arm.c", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786484, i32 0, null, metadata !"x", metadata !"x", metadata !"", metadata !5, i32 1, metadata !6, i32 0, i32 1, i32* @x, null} ; [ DW_TAG_variable ] [x] [line 1] [def]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/arm.c]
!6 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
