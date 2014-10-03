; RUN: llc %s -o /dev/null
; Here variable bar is optimized away. Do not trip over while trying to generate debug info.


define i32 @foo() nounwind uwtable readnone ssp {
entry:
  ret i32 42, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 139632)\001\00\000\00\000", metadata !17, metadata !1, metadata !1, metadata !3, metadata !12, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\000\001\000", metadata !17, metadata !6, metadata !7, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 0] [foo]
!6 = metadata !{metadata !"0x29", metadata !17} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !14}
!14 = metadata !{metadata !"0x34\00bar\00bar\00\002\001\001", metadata !5, metadata !6, metadata !9, null, null} ; [ DW_TAG_variable ]
!15 = metadata !{i32 3, i32 3, metadata !16, null}
!16 = metadata !{metadata !"0xb\001\0011\000", metadata !17, metadata !5} ; [ DW_TAG_lexical_block ]
!17 = metadata !{metadata !"fb.c", metadata !"/private/tmp"}
!18 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
