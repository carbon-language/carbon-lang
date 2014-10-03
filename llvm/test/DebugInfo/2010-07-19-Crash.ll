; RUN: llc -o /dev/null < %s
; PR7662
; Do not add variables to !11 because it is a declaration entry.

define i32 @bar() nounwind readnone ssp {
entry:
  ret i32 42, !dbg !9
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15}
!llvm.dbg.sp = !{!0, !6, !11}
!llvm.dbg.lv.foo = !{!7}

!0 = metadata !{metadata !"0x2e\00bar\00bar\00bar\003\000\001\000\006\000\001\000", metadata !12, metadata !1, metadata !3, null, i32 ()* @bar, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !12} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang 2.8\001\00\000\00\000", metadata !12, metadata !14, metadata !14, metadata !13, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !12, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !12, metadata !1} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00foo\00foo\00foo\007\001\001\000\006\000\001\000", metadata !12, metadata !1, metadata !3, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!7 = metadata !{metadata !"0x100\00one\008\000", metadata !8, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!8 = metadata !{metadata !"0xb\007\0018\000", metadata !12, metadata !6} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 4, i32 3, metadata !10, null}
!10 = metadata !{metadata !"0xb\003\0011\000", metadata !12, metadata !0} ; [ DW_TAG_lexical_block ]
!11 = metadata !{metadata !"0x2e\00foo\00foo\00foo\007\001\000\000\006\000\001\000", metadata !12, metadata !1, metadata !3, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!12 = metadata !{metadata !"one.c", metadata !"/private/tmp"}
!13 = metadata !{metadata !0}
!14 = metadata !{i32 0}
!15 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
