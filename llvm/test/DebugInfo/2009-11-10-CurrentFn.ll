; RUN: llc < %s -o /dev/null

define void @bar(i32 %i) nounwind uwtable ssp {
entry:
  tail call void (...)* @foo() nounwind, !dbg !14
  ret void, !dbg !16
}

declare void @foo(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 139632)\001\00\000\00\000", metadata !17, metadata !1, metadata !1, metadata !3, metadata !1, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00bar\00bar\00\003\000\001\000\006\00256\001\000", metadata !17, metadata !6, metadata !7, null, void (i32)* @bar, null, null, metadata !9} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 0] [bar]
!6 = metadata !{metadata !"0x29", metadata !17} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{metadata !11}
!11 = metadata !{metadata !"0x101\00i\0016777219\000", metadata !17, metadata !5, metadata !12} ; [ DW_TAG_arg_variable ]
!12 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!13 = metadata !{i32 3, i32 14, metadata !5, null}
!14 = metadata !{i32 4, i32 3, metadata !15, null}
!15 = metadata !{metadata !"0xb\003\0017\000", metadata !17, metadata !5} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 5, i32 1, metadata !15, null}
!17 = metadata !{metadata !"cf.c", metadata !"/private/tmp"}
!18 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
