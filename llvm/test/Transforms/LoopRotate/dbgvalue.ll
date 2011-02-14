; RUN: opt -S -loop-rotate  %s  | FileCheck %s

; CHECK: entry
; CHECK-NEXT: call void @llvm.dbg.value(metadata !{i32 %x}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @tak(i32 %x, i32 %y, i32 %z) nounwind ssp {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then, %entry
  %x.tr = phi i32 [ %x, %entry ], [ %call, %if.then ]
  %y.tr = phi i32 [ %y, %entry ], [ %call9, %if.then ]
  %z.tr = phi i32 [ %z, %entry ], [ %call14, %if.then ]
  tail call void @llvm.dbg.value(metadata !{i32 %x.tr}, i64 0, metadata !6), !dbg !7
  tail call void @llvm.dbg.value(metadata !{i32 %y.tr}, i64 0, metadata !8), !dbg !9
  tail call void @llvm.dbg.value(metadata !{i32 %z.tr}, i64 0, metadata !10), !dbg !11
  %cmp = icmp slt i32 %y.tr, %x.tr, !dbg !12
  br i1 %cmp, label %if.then, label %if.end, !dbg !12

if.then:                                          ; preds = %tailrecurse
  %sub = sub nsw i32 %x.tr, 1, !dbg !14
  %call = tail call i32 @tak(i32 %sub, i32 %y.tr, i32 %z.tr), !dbg !14
  %sub6 = sub nsw i32 %y.tr, 1, !dbg !14
  %call9 = tail call i32 @tak(i32 %sub6, i32 %z.tr, i32 %x.tr), !dbg !14
  %sub11 = sub nsw i32 %z.tr, 1, !dbg !14
  %call14 = tail call i32 @tak(i32 %sub11, i32 %x.tr, i32 %y.tr), !dbg !14
  br label %tailrecurse

if.end:                                           ; preds = %tailrecurse
  br label %return, !dbg !16

return:                                           ; preds = %if.end
  ret i32 %z.tr, !dbg !17
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"tak", metadata !"tak", metadata !"", metadata !1, i32 32, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 (i32, i32, i32)* @tak} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"/Volumes/Lalgate/cj/llvm/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame/recursive.c", metadata !"/Volumes/Lalgate/cj/D/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"/Volumes/Lalgate/cj/llvm/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame/recursive.c", metadata !"/Volumes/Lalgate/cj/D/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame", metadata !"clang version 2.9 (trunk 125492)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 589860, metadata !2, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 590081, metadata !0, metadata !"x", metadata !1, i32 32, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 32, i32 13, metadata !0, null}
!8 = metadata !{i32 590081, metadata !0, metadata !"y", metadata !1, i32 32, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 32, i32 20, metadata !0, null}
!10 = metadata !{i32 590081, metadata !0, metadata !"z", metadata !1, i32 32, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 32, i32 27, metadata !0, null}
!12 = metadata !{i32 33, i32 3, metadata !13, null}
!13 = metadata !{i32 589835, metadata !0, i32 32, i32 30, metadata !1, i32 6} ; [ DW_TAG_lexical_block ]
!14 = metadata !{i32 34, i32 5, metadata !15, null}
!15 = metadata !{i32 589835, metadata !13, i32 33, i32 14, metadata !1, i32 7} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 36, i32 3, metadata !13, null}
!17 = metadata !{i32 37, i32 1, metadata !13, null}
