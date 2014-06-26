; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; struct b {
;   double **b;
; };
;
; void a(struct b *A) {
;   for (int i=0; i<32; i++)
;     A->b[i] = 0;
; }

; CHECK: remark: ReportVariantBasePtr01.c:6:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportVariantBasePtr01.c:7:5: The base address of this array is not invariant inside the loop

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.b = type { double** }

define void @a(%struct.b* nocapture readonly %A) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata !{%struct.b* %A}, i64 0, metadata !16), !dbg !23
  tail call void @llvm.dbg.value(metadata !24, i64 0, metadata !17), !dbg !25
  %b = getelementptr inbounds %struct.b* %A, i64 0, i32 0, !dbg !26
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body, %entry.split
  %indvar4 = phi i64 [ %indvar.next, %for.body ], [ 0, %entry.split ]
  %0 = mul i64 %indvar4, 4, !dbg !26
  %1 = add i64 %0, 3, !dbg !26
  %2 = add i64 %0, 2, !dbg !26
  %3 = add i64 %0, 1, !dbg !26
  %4 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx = getelementptr double** %4, i64 %0, !dbg !26
  store double* null, double** %arrayidx, align 8, !dbg !26, !tbaa !33
  %5 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.1 = getelementptr double** %5, i64 %3, !dbg !26
  store double* null, double** %arrayidx.1, align 8, !dbg !26, !tbaa !33
  %6 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.2 = getelementptr double** %6, i64 %2, !dbg !26
  store double* null, double** %arrayidx.2, align 8, !dbg !26, !tbaa !33
  %7 = load double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.3 = getelementptr double** %7, i64 %1, !dbg !26
  store double* null, double** %arrayidx.3, align 8, !dbg !26, !tbaa !33
  %indvar.next = add i64 %indvar4, 1, !dbg !27
  %exitcond = icmp eq i64 %indvar.next, 8, !dbg !27
  br i1 %exitcond, label %for.end, label %for.body, !dbg !27

for.end:                                          ; preds = %for.body
  ret void, !dbg !34
}

declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [test/ScopDetectionDiagnostics/ReportVariantBasePtr01.c] [DW_LANG_C99]
!1 = metadata !{metadata !"ReportVariantBasePtr01.c", metadata !"test/ScopDetectionDiagnostics"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"a", metadata !"a", metadata !"", i32 5, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (%struct.b*)* @a, null, null, metadata !15, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [a]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [test/ScopDetectionDiagnostics/ReportVariantBasePtr01.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from b]
!9 = metadata !{i32 786451, metadata !1, null, metadata !"b", i32 1, i64 64, i64 64, i32 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [b] [line 1, size 64, align 64, offset 0] [def] [from ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"b", i32 2, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ] [b] [line 2, size 64, align 64, offset 0] [from ]
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from double]
!14 = metadata !{i32 786468, null, null, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!15 = metadata !{metadata !16, metadata !17}
!16 = metadata !{i32 786689, metadata !4, metadata !"A", metadata !5, i32 16777221, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [A] [line 5]
!17 = metadata !{i32 786688, metadata !18, metadata !"i", metadata !5, i32 6, metadata !19, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [i] [line 6]
!18 = metadata !{i32 786443, metadata !1, metadata !4, i32 6, i32 3, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [test/ScopDetectionDiagnostics/ReportVariantBasePtr01.c]
!19 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!20 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!21 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!22 = metadata !{metadata !"clang version 3.5.0 "}
!23 = metadata !{i32 5, i32 18, metadata !4, null}
!24 = metadata !{i32 0}
!25 = metadata !{i32 6, i32 12, metadata !18, null}
!26 = metadata !{i32 7, i32 5, metadata !18, null}
!27 = metadata !{i32 6, i32 8, metadata !18, null}
!28 = metadata !{metadata !29, metadata !30, i64 0}
!29 = metadata !{metadata !"b", metadata !30, i64 0}
!30 = metadata !{metadata !"any pointer", metadata !31, i64 0}
!31 = metadata !{metadata !"omnipotent char", metadata !32, i64 0}
!32 = metadata !{metadata !"Simple C/C++ TBAA"}
!33 = metadata !{metadata !30, metadata !30, i64 0}
!34 = metadata !{i32 8, i32 1, metadata !4, null} ; [ DW_TAG_imported_declaration ]
