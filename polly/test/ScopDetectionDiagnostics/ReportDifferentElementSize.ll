; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; 1 void differenttypes(char *A)
; 2 {
; 3   for (long i = 0; i < 1024; ++i)
; 4     ((float*)A)[i] = ((double*)A)[i];
; 5 }

; CHECK: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; CHECK-NEXT: remark: /tmp/test.c:4:14: The array "A" is accessed through elements that differ in size
; CHECK-NEXT: remark: /tmp/test.c:4:32: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @differenttypes(i8* nocapture %A)  {
entry:
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.body, %entry
  %i.05 = phi i64 [ 0, %entry ], [ %tmp11, %for.body ]
  %tmp = shl i64 %i.05, 3, !dbg !15
  %uglygep = getelementptr i8* %A, i64 %tmp
  %arrayidx = bitcast i8* %uglygep to double*, !dbg !16
  %tmp9 = shl i64 %i.05, 2, !dbg !15
  %uglygep7 = getelementptr i8* %A, i64 %tmp9
  %arrayidx1 = bitcast i8* %uglygep7 to float*, !dbg !17
  %tmp10 = load double* %arrayidx, align 8, !dbg !16, !tbaa !18
  %conv = fptrunc double %tmp10 to float, !dbg !16
  store float %conv, float* %arrayidx1, align 4, !dbg !17, !tbaa !22
  %tmp11 = add nsw i64 %i.05, 1, !dbg !24
  %exitcond = icmp eq i64 %tmp11, 1024, !dbg !10
  br i1 %exitcond, label %for.end, label %for.body, !dbg !10

for.end:                                          ; preds = %for.body
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2} ; [ DW_TAG_compile_unit ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"/tmp/test.c", metadata !"/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"differenttypes", metadata !"differenttypes", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i8*)* @differenttypes, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [differenttypes]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.6.0 "}
!10 = metadata !{i32 3, i32 20, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !12, i32 2} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!12 = metadata !{i32 786443, metadata !1, metadata !13, i32 1} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!13 = metadata !{i32 786443, metadata !1, metadata !14, i32 3, i32 3, i32 1} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!14 = metadata !{i32 786443, metadata !1, metadata !4, i32 3, i32 3, i32 0} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test/ScopDetectionDiagnostics//tmp/test.c]
!15 = metadata !{i32 4, i32 32, metadata !13, null}
!16 = metadata !{i32 4, i32 22, metadata !13, null}
!17 = metadata !{i32 4, i32 14, metadata !13, null}
!18 = metadata !{metadata !19, metadata !19, i64 0}
!19 = metadata !{metadata !"double", metadata !20, i64 0}
!20 = metadata !{metadata !"omnipotent char", metadata !21, i64 0}
!21 = metadata !{metadata !"Simple C/C++ TBAA"}
!22 = metadata !{metadata !23, metadata !23, i64 0}
!23 = metadata !{metadata !"float", metadata !20, i64 0}
!24 = metadata !{i32 3, i32 30, metadata !13, null}
!25 = metadata !{i32 5, i32 1, metadata !4, null}
