; RUN: opt -strip-dead-debug-info -disable-output < %s
define i32 @foo() nounwind ssp {
entry:
  ret i32 0, !dbg !8
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00foo\003\000\001\000\006\000\000\000", metadata !10, metadata !1, metadata !3, null, i32 ()* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !10} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.8 (trunk 112062)\001\00\000\00\001", metadata !10, metadata !11, metadata !11, metadata !12, metadata !13, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !10, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !10, metadata !1} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x34\00i\00i\00i\002\001\001", metadata !1, metadata !1, metadata !7, i32 0, null} ; [ DW_TAG_variable ]
!7 = metadata !{metadata !"0x26\00\000\000\000\000\000", metadata !10, metadata !1, metadata !5} ; [ DW_TAG_const_type ]
!8 = metadata !{i32 3, i32 13, metadata !9, null}
!9 = metadata !{metadata !"0xb\003\0011\000", metadata !10, metadata !0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"/tmp/a.c", metadata !"/Volumes/Lalgate/clean/D.CW"}
!11 = metadata !{i32 0}
!12 = metadata !{metadata !0}
!13 = metadata !{metadata !6}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
