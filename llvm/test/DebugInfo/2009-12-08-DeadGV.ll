; RUN: llc  %s -o /dev/null
; PR 5713

define i32 @_Z3foov() nounwind readnone ssp {
entry:
  ret i32 0, !dbg !4
}

!llvm.dbg.gv = !{!0}

!0 = metadata !{i32 458804, i32 0, metadata !1, metadata !"X", metadata !"X", metadata !"_ZN1A1XE", metadata !2, i32 3, metadata !3, i1 false, i1 true, null}; [DW_TAG_variable ]
!1 = metadata !{i32 458809, metadata !2, metadata !"A", metadata !2, i32 2}; [DW_TAG_namespace ]
!2 = metadata !{i32 458769, i32 0, i32 4, metadata !"ng.cc", metadata !"/tmp", metadata !"4.2.1 (Based on Apple Inc. build 5653) (LLVM build)", i1 true, i1 true, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!3 = metadata !{i32 458788, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}; [DW_TAG_base_type ]
!4 = metadata !{i32 7, i32 0, metadata !5, null}
!5 = metadata !{i32 458798, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"_Z3foov", metadata !2, i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, null}; [DW_TAG_subprogram ]
!6 = metadata !{i32 458773, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0}; [DW_TAG_subroutine_type ]
!7 = metadata !{metadata !3}
