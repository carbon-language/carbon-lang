; RUN: opt -analyze -module-debuginfo < %s | FileCheck %s

; Produced from linking:
;   /tmp/test1.c containing f()
;   /tmp/test2.c containing g()

; Verify that both compile units and both their contained functions are
; listed by DebugInfoFinder:
;CHECK: Compile Unit: [ DW_TAG_compile_unit ] [/tmp/test1.c] [DW_LANG_C99]
;CHECK: Compile Unit: [ DW_TAG_compile_unit ] [/tmp/test2.c] [DW_LANG_C99]
;CHECK: Subprogram: [ DW_TAG_subprogram ] [line 1] [def] [f]
;CHECK: Subprogram: [ DW_TAG_subprogram ] [line 1] [def] [g]

define void @f() {
  ret void, !dbg !14
}

define void @g() {
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0, !8}
!llvm.module.flags = !{!13, !16}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.4 (192092)\000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/test1.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test1.c", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00f\00f\00\001\000\001\000\006\000\000\001", metadata !1, metadata !5, metadata !6, null, void ()* @f, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/test1.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !"0x11\0012\00clang version 3.4 (192092)\000\00\000\00\000", metadata !9, metadata !2, metadata !2, metadata !10, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/test2.c] [DW_LANG_C99]
!9 = metadata !{metadata !"test2.c", metadata !"/tmp"}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"0x2e\00g\00g\00\001\000\001\000\006\000\000\001", metadata !9, metadata !12, metadata !6, null, void ()* @g, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [g]
!12 = metadata !{metadata !"0x29", metadata !9}         ; [ DW_TAG_file_type ] [/tmp/test2.c]
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!14 = metadata !{i32 1, i32 0, metadata !4, null}
!15 = metadata !{i32 1, i32 0, metadata !11, null}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
