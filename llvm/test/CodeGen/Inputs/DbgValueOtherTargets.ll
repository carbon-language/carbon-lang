; Check that DEBUG_VALUE comments come through on a variety of targets.

define i32 @main() nounwind ssp {
entry:
; CHECK: DEBUG_VALUE
  call void @llvm.dbg.value(metadata !6, i64 0, metadata !7, metadata !{metadata !"0x102"}), !dbg !9
  ret i32 0, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}

!0 = metadata !{metadata !"0x2e\00main\00main\00\002\000\001\000\006\000\000\000", metadata !12, metadata !1, metadata !3, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !12} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 120996)\000\00\000\00\000", metadata !12, metadata !6, metadata !6, metadata !11, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !12, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !12, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 0}
!7 = metadata !{metadata !"0x100\00i\003\000", metadata !8, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!8 = metadata !{metadata !"0xb\002\0012\000", metadata !12, metadata !0} ; [ DW_TAG_lexical_block ]
!9 = metadata !{i32 3, i32 11, metadata !8, null}
!10 = metadata !{i32 4, i32 2, metadata !8, null}
!11 = metadata !{metadata !0}
!12 = metadata !{metadata !"/tmp/x.c", metadata !"/Users/manav"}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
