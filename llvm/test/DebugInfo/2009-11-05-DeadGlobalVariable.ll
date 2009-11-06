; RUN: llc %s -o /dev/null
; Here variable bar is optimzied away. Do not trip over while trying to generate debug info.

define i32 @foo() nounwind readnone optsize ssp {
entry:
  ret i32 42, !dbg !6
}

!llvm.dbg.gv = !{!0}

!0 = metadata !{i32 458804, i32 0, metadata !1, metadata !"foo.bar", metadata !"foo.bar", metadata !"foo.bar", metadata !2, i32 3, metadata !5, i1 true, i1 true, null}; [DW_TAG_variable ]
!1 = metadata !{i32 458798, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"foo", metadata !2, i32 2, metadata !3, i1 false, i1 true}; [DW_TAG_subprogram ]
!2 = metadata !{i32 458769, i32 0, i32 12, metadata !"st.c", metadata !"/private/tmp", metadata !"clang 1.1", i1 true, i1 true, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!3 = metadata !{i32 458773, metadata !2, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0}; [DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 458788, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}; [DW_TAG_base_type ]
!6 = metadata !{i32 5, i32 1, metadata !1, null}
