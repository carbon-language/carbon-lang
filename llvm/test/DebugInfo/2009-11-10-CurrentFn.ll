; RUN: llc < %s -o /dev/null

declare void @foo()

define void @bar(i32 %i) nounwind ssp {
entry:
  tail call void @foo() nounwind, !dbg !0
  ret void, !dbg !6
}

!0 = metadata !{i32 9, i32 0, metadata !1, null}
!1 = metadata !{i32 458798, i32 0, metadata !2, metadata !"baz", metadata !"baz", metadata !"baz", metadata !2, i32 8, metadata !3, i1 true, i1 true}; [DW_TAG_subprogram ]
!2 = metadata !{i32 458769, i32 0, i32 1, metadata !"2007-12-VarArrayDebug.c", metadata !"/Volumes/Data/ddunbar/llvm/test/FrontendC", metadata !"4.2.1 (Based on Apple Inc. build 5653) (LLVM build)", i1 true, i1 true, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!3 = metadata !{i32 458773, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0}; [DW_TAG_subroutine_type ]
!4 = metadata !{null, metadata !5}
!5 = metadata !{i32 458788, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}; [DW_TAG_base_type ]
!6 = metadata !{i32 18, i32 0, metadata !7, null}
!7 = metadata !{i32 458798, i32 0, metadata !2, metadata !"bar", metadata !"bar", metadata !"bar", metadata !2, i32 16, metadata !8, i1 false, i1 true}; [DW_TAG_subprogram ]
!8 = metadata !{i32 458773, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !9, i32 0}; [DW_TAG_subroutine_type ]
!9 = metadata !{null}
