; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s
target triple = "dxil-unknown-unknown"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; CHECK: define float @fma(float, float, float) unnamed_addr #0 !dbg !7
; Function Attrs: norecurse nounwind readnone willreturn
define dso_local float @fma(float %0, float %1, float %2) local_unnamed_addr #0 !dbg !6 {
; CHECK-NEXT: call void @llvm.dbg.value(metadata float %0, metadata !12, metadata !15), !dbg !16
; CHECK-NEXT: call void @llvm.dbg.value(metadata float %1, metadata !13, metadata !15), !dbg !16
; CHECK-NEXT: call void @llvm.dbg.value(metadata float %2, metadata !14, metadata !15), !dbg !16
  call void @llvm.dbg.value(metadata float %0, metadata !11, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata float %1, metadata !12, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata float %2, metadata !13, metadata !DIExpression()), !dbg !14
; CHECK-NEXT: %4 = fmul float %0, %1, !dbg !17
; CHECK-NEXT: %5 = fadd float %4, %2, !dbg !18
  %4 = fmul float %0, %1, !dbg !15
  %5 = fadd float %4, %2, !dbg !16
  ret float %5, !dbg !17
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind readnone willreturn }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

; Other tests verify that we come back with reasonable structure for the debug
; info types, this test just needs to ensure they are there.
; The patch this is paired with fixes a bug where function debug info wasn't
; being emitted correctly even though other tests verified the MD would be
; emitted if it was referenced as module metadata.

; CHECK:      !0 = distinct !DICompileUnit
; CHECK-NEXT: !1 = !DIFile(filename:
; CHECK:      !7 = distinct !DISubprogram(name: "fma",
; CHECK:      !12 = !DILocalVariable(tag:
; CHECK-NEXT: !13 = !DILocalVariable(tag:
; CHECK-NEXT: !14 = !DILocalVariable(tag:
; CHECK-NEXT: !15 = !DIExpression()
; CHECK-NEXT: !16 = !DILocation(line:
; CHECK-NEXT: !17 = !DILocation(line:
; CHECK-NEXT: !18 = !DILocation(line:
; CHECK-NEXT: !19 = !DILocation(line:

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "in.c", directory: "dir")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"Some Compiler"}
!6 = distinct !DISubprogram(name: "fma", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !9, !9}
!9 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!10 = !{!11, !12, !13}
!11 = !DILocalVariable(name: "x", arg: 1, scope: !6, file: !1, line: 1, type: !9)
!12 = !DILocalVariable(name: "y", arg: 2, scope: !6, file: !1, line: 1, type: !9)
!13 = !DILocalVariable(name: "z", arg: 3, scope: !6, file: !1, line: 1, type: !9)
!14 = !DILocation(line: 0, scope: !6)
!15 = !DILocation(line: 2, column: 12, scope: !6)
!16 = !DILocation(line: 2, column: 16, scope: !6)
!17 = !DILocation(line: 2, column: 3, scope: !6)
