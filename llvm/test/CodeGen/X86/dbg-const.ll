; RUN: llc < %s - | FileCheck %s
target triple = "x86_64-apple-darwin10.0.0"

;CHECK:        ## DW_OP_constu
;CHECK-NEXT:  .byte	42
define i32 @foobar() nounwind readonly noinline ssp {
entry:
  %call = tail call i32 @bar(), !dbg !11
  tail call void @llvm.dbg.value(metadata !8, i64 0, metadata !6), !dbg !9
  %call2 = tail call i32 @bar(), !dbg !11
  tail call void @llvm.dbg.value(metadata !{i32 %call}, i64 0, metadata !6), !dbg !11
  %add = add nsw i32 %call2, %call, !dbg !12
  ret i32 %add, !dbg !10
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone
declare i32 @bar() nounwind readnone

!llvm.dbg.cu = !{!2}

!0 = metadata !{i32 786478, i32 0, metadata !1, metadata !"foobar", metadata !"foobar", metadata !"foobar", metadata !1, i32 12, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 true, i32 ()* @foobar, null, null, metadata !14, i32 0}
!1 = metadata !{i32 786473, metadata !15} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, i32 0, i32 12, metadata !1, metadata !"clang version 2.9 (trunk 114183)", i1 true, metadata !"", i32 0, null, null, metadata !13, null, metadata !""}
!3 = metadata !{i32 786453, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!6 = metadata !{i32 786688, metadata !7, metadata !"j", metadata !1, i32 15, metadata !5, i32 0, null}
!7 = metadata !{i32 786443, metadata !0, i32 12, i32 52, metadata !1, i32 0}
!8 = metadata !{i32 42}
!9 = metadata !{i32 15, i32 12, metadata !7, null}
!10 = metadata !{i32 23, i32 3, metadata !7, null}
!11 = metadata !{i32 17, i32 3, metadata !7, null}
!12 = metadata !{i32 18, i32 3, metadata !7, null}
!13 = metadata !{metadata !0}
!14 = metadata !{metadata !6}
!15 = metadata !{metadata !"mu.c", metadata !"/private/tmp"}
