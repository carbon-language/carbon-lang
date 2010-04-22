; RUN: llc < %s | grep "Ltmp3-Lfunc_begin"
; PR 6894

declare void @_Z7examplev() ssp

define linkonce_odr void @_bar(i64, i64, i64 %__depth_limit) ssp {
entry:
  br i1 undef, label %while.body, label %while.end, !dbg !0

while.body:                                       ; preds = %entry
  br i1 undef, label %if.then, label %if.end, !dbg !8

if.then:                                          ; preds = %while.body
  call void @_Z7examplev(), !dbg !10
  ret void, !dbg !12

if.end:                                           ; preds = %while.body
  call void @_Z7examplev(), !dbg !13
  unreachable

while.end:                                        ; preds = %entry
  ret void, !dbg !12
}

!0 = metadata !{i32 2742, i32 7, metadata !1, null}
!1 = metadata !{i32 524299, metadata !2, i32 2738, i32 5} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 524334, i32 0, metadata !3, metadata !"__introsort_loop", metadata !"__introsort_loop", metadata !"_bar", metadata !3, i32 2738, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!3 = metadata !{i32 524329, metadata !"stl_algo.h", metadata !"/usr/include/c++/4.2.1/bits", metadata !4} ; [ DW_TAG_file_type ]
!4 = metadata !{i32 524305, i32 0, i32 4, metadata !"example.cc", metadata !"/tmp", metadata !"clang 1.5", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 524309, metadata !6, metadata !"", metadata !6, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null} ; [ DW_TAG_subroutine_type ]
!6 = metadata !{i32 524329, metadata !"example.cc", metadata !"/tmp", metadata !4} ; [ DW_TAG_file_type ]
!7 = metadata !{null}
!8 = metadata !{i32 2744, i32 4, metadata !9, null}
!9 = metadata !{i32 524299, metadata !1, i32 2743, i32 2} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 2746, i32 8, metadata !11, null}
!11 = metadata !{i32 524299, metadata !9, i32 2745, i32 6} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 2762, i32 5, metadata !1, null}
!13 = metadata !{i32 2750, i32 4, metadata !9, null}
