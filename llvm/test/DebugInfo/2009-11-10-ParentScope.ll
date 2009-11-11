; RUN: llc < %s -o /dev/null
%struct.htab = type { i32 (i8*)*, i32 (i8*, i8*)*, void (i8*)*, i8**, i64, i64, i64, i32, i32, i8* (i64, i64)*, void (i8*)*, i8*, i8* (i8*, i64, i64)*, void (i8*, i8*)*, i32, [4 x i8] }

define i8* @htab_find_with_hash(%struct.htab* %htab, i8* %element, i32 %hash) nounwind {
entry:
  br i1 undef, label %land.lhs.true, label %if.end, !dbg !0

land.lhs.true:                                    ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  store i8* undef, i8** undef, !dbg !7
  ret i8* undef, !dbg !10
}

!0 = metadata !{i32 571, i32 3, metadata !1, null}
!1 = metadata !{i32 458763, metadata !2}; [DW_TAG_lexical_block ]
!2 = metadata !{i32 458798, i32 0, metadata !3, metadata !"htab_find_with_hash", metadata !"htab_find_with_hash", metadata !"htab_find_with_hash", metadata !3, i32 561, metadata !4, i1 false, i1 true}; [DW_TAG_subprogram ]
!3 = metadata !{i32 458769, i32 0, i32 12, metadata !"hashtab.c", metadata !"/usr/src/gnu/usr.bin/cc/cc_tools/../../../../contrib/gcclibs/libiberty", metadata !"clang 1.1", i1 true, i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!4 = metadata !{i32 458773, metadata !3, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0}; [DW_TAG_subroutine_type ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 458767, metadata !3, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, null}; [DW_TAG_pointer_type ]
!7 = metadata !{i32 583, i32 7, metadata !8, null}
!8 = metadata !{i32 458763, metadata !9}; [DW_TAG_lexical_block ]
!9 = metadata !{i32 458763, metadata !1}; [DW_TAG_lexical_block ]
!10 = metadata !{i32 588, i32 1, metadata !2, null}
