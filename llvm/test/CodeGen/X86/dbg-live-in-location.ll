; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

@str = internal constant [3 x i8] c"Hi\00"

define void @foo() nounwind ssp {
entry:
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([3 x i8]* @str, i64 0, i64 0))
  ret void, !dbg !17
}

; CHECK: arg.c:5:14

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !9), !dbg !19
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !10), !dbg !20
  %cmp = icmp sgt i32 %argc, 1, !dbg !21
  br i1 %cmp, label %cond.end, label %for.body.lr.ph, !dbg !21

cond.end:                                         ; preds = %entry
  %arrayidx = getelementptr inbounds i8** %argv, i64 1, !dbg !21
  %tmp2 = load i8** %arrayidx, align 8, !dbg !21, !tbaa !22
  %call = tail call i32 (...)* @atoi(i8* %tmp2) nounwind, !dbg !21
  tail call void @llvm.dbg.value(metadata !{i32 %call}, i64 0, metadata !16), !dbg !21
  tail call void @llvm.dbg.value(metadata !25, i64 0, metadata !14), !dbg !26
  %cmp57 = icmp sgt i32 %call, 0, !dbg !26
  br i1 %cmp57, label %for.body.lr.ph, label %for.end, !dbg !26

for.body.lr.ph:                                   ; preds = %entry, %cond.end
  %cond10 = phi i32 [ %call, %cond.end ], [ 300, %entry ]
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %i.08 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %puts.i = tail call i32 @puts(i8* getelementptr inbounds ([3 x i8]* @str, i64 0, i64 0)) nounwind
  %inc = add nsw i32 %i.08, 1, !dbg !27
  %exitcond = icmp eq i32 %inc, %cond10
  br i1 %exitcond, label %for.end, label %for.body, !dbg !26

for.end:                                          ; preds = %for.body, %cond.end
  ret i32 0, !dbg !29
}

declare i32 @atoi(...)

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.sp = !{!0, !5}
!llvm.dbg.lv.main = !{!9, !10, !14, !16}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"", metadata !1, i32 2, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 true, void ()* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"arg.c", metadata !"/private/tmp", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"arg.c", metadata !"/private/tmp", metadata !"clang version 2.9 (trunk 124504)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 589870, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"", metadata !1, i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32 (i32, i8**)* @main} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 589860, metadata !2, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 590081, metadata !5, metadata !"argc", metadata !1, i32 5, metadata !8, i32 0} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 590081, metadata !5, metadata !"argv", metadata !1, i32 5, metadata !11, i32 0} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 589839, metadata !2, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{i32 589839, metadata !2, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_pointer_type ]
!13 = metadata !{i32 589860, metadata !2, metadata !"char", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 590080, metadata !15, metadata !"i", metadata !1, i32 7, metadata !8, i32 0} ; [ DW_TAG_auto_variable ]
!15 = metadata !{i32 589835, metadata !5, i32 6, i32 1, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 590080, metadata !15, metadata !"iterations", metadata !1, i32 8, metadata !8, i32 0} ; [ DW_TAG_auto_variable ]
!17 = metadata !{i32 4, i32 1, metadata !18, null}
!18 = metadata !{i32 589835, metadata !0, i32 2, i32 12, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 5, i32 14, metadata !5, null}
!20 = metadata !{i32 5, i32 26, metadata !5, null}
!21 = metadata !{i32 8, i32 51, metadata !15, null}
!22 = metadata !{metadata !"any pointer", metadata !23}
!23 = metadata !{metadata !"omnipotent char", metadata !24}
!24 = metadata !{metadata !"Simple C/C++ TBAA", null}
!25 = metadata !{i32 0}
!26 = metadata !{i32 9, i32 2, metadata !15, null}
!27 = metadata !{i32 9, i32 30, metadata !28, null}
!28 = metadata !{i32 589835, metadata !15, i32 9, i32 2, metadata !1, i32 2} ; [ DW_TAG_lexical_block ]
!29 = metadata !{i32 12, i32 9, metadata !15, null}
