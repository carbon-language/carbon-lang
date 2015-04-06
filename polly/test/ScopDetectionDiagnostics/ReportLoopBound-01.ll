; RUN: opt %loadPolly -polly-detect-unprofitable -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; void f(int A[], int n) {
;   for (int i = 0; i < A[n]; i++)
;     A[i] = 0;
; }

; CHECK: remark: ReportLoopBound-01.c:2:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportLoopBound-01.c:2:8: Failed to derive an affine function from the loop bounds.
; CHECK: remark: ReportLoopBound-01.c:3:5: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i32* %A, i32 %n) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32* %A, i64 0, metadata !13, metadata !MDExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata i32* %A, i64 0, metadata !13, metadata !MDExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata i32 %n, i64 0, metadata !15, metadata !MDExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !18, metadata !MDExpression()), !dbg !20
  %idxprom = sext i32 %n to i64, !dbg !21
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom, !dbg !21
  %0 = load i32, i32* %arrayidx, align 4, !dbg !21
  %cmp3 = icmp sgt i32 %0, 0, !dbg !21
  br i1 %cmp3, label %for.body.lr.ph, label %for.end, !dbg !21

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body, !dbg !22

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %arrayidx2 = getelementptr i32, i32* %A, i64 %indvar, !dbg !24
  %1 = add i64 %indvar, 1, !dbg !24
  %inc = trunc i64 %1 to i32, !dbg !21
  store i32 0, i32* %arrayidx2, align 4, !dbg !24
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !18, metadata !MDExpression()), !dbg !20
  %2 = load i32, i32* %arrayidx, align 4, !dbg !21
  %cmp = icmp slt i32 %inc, %2, !dbg !21
  %indvar.next = add i64 %indvar, 1, !dbg !21
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !dbg !21

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end, !dbg !25

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !MDFile(filename: "ReportLoopBound-01.c", directory: "test/ScopDetectionDiagnostic/")
!2 = !{}
!3 = !{!4}
!4 = !MDSubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !1, scope: !5, type: !6, function: void (i32*, i32)* @f, variables: !2)
!5 = !MDFile(filename: "ReportLoopBound-01.c", directory: "test/ScopDetectionDiagnostic/")
!6 = !MDSubroutineType(types: !7)
!7 = !{null, !8, !9}
!8 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.6.0 "}
!13 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "A", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!14 = !MDLocation(line: 1, column: 12, scope: !4)
!15 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "n", line: 1, arg: 2, scope: !4, file: !5, type: !9)
!16 = !MDLocation(line: 1, column: 21, scope: !4)
!17 = !{i32 0}
!18 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "i", line: 2, scope: !19, file: !5, type: !9)
!19 = distinct !MDLexicalBlock(line: 2, column: 3, file: !1, scope: !4)
!20 = !MDLocation(line: 2, column: 12, scope: !19)
!21 = !MDLocation(line: 2, column: 8, scope: !19)
!22 = !MDLocation(line: 2, column: 8, scope: !23)
!23 = distinct !MDLexicalBlock(line: 2, column: 8, file: !1, scope: !19)
!24 = !MDLocation(line: 3, column: 5, scope: !19)
!25 = !MDLocation(line: 2, column: 8, scope: !26)
!26 = distinct !MDLexicalBlock(line: 2, column: 8, file: !1, scope: !19)
!27 = !MDLocation(line: 4, column: 1, scope: !4)
