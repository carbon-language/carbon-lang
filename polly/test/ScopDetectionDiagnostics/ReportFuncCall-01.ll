; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1 | FileCheck %s

; #define N 1024
; double invalidCall(double A[N]);
;
; void a(double A[N], int n) {
;   for (int i=0; i<n; ++i) {
;     A[i] = invalidCall(A);
;   }
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @a(double* %A, i32 %n) #0 {
entry:
  %cmp1 = icmp sgt i32 %n, 0, !dbg !10
  br i1 %cmp1, label %for.body.lr.ph, label %for.end, !dbg !10

for.body.lr.ph:                                   ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr double* %A, i64 %indvar, !dbg !12
  %call = tail call double @invalidCall(double* %A) #2, !dbg !12
  store double %call, double* %arrayidx, align 8, !dbg !12, !tbaa !14
  %indvar.next = add i64 %indvar, 1, !dbg !10
  %exitcond = icmp eq i64 %indvar.next, %0, !dbg !10
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !10

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !18
}

declare double @invalidCall(double*) #1

; CHECK: remark: ReportFuncCall.c:4:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportFuncCall.c:5:12: This function call cannot be handeled. Try to inline it.

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2} ; [ DW_TAG_compile_unit ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c] [DW_LANG_C99]
!1 = metadata !{metadata !"ReportFuncCall.c", metadata !"/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"a", metadata !"a", metadata !"", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (double*, i32)* @a, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [a]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.5.0 "}
!10 = metadata !{i32 4, i32 8, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !4, i32 4, i32 3, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c]
!12 = metadata !{i32 5, i32 12, metadata !13, null}
!13 = metadata !{i32 786443, metadata !1, metadata !11, i32 4, i32 27, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c]
!14 = metadata !{metadata !15, metadata !15, i64 0}
!15 = metadata !{metadata !"double", metadata !16, i64 0}
!16 = metadata !{metadata !"omnipotent char", metadata !17, i64 0}
!17 = metadata !{metadata !"Simple C/C++ TBAA"}
!18 = metadata !{i32 7, i32 1, metadata !4, null}
