; RUN: llc -O0 -stop-before=livedebugvalues < %s | FileCheck %s

; ModuleID = '/tmp/t.o'
source_filename = "/tmp/t.o"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx11.0.0"

define swiftcc void @"$s1t1f1bySb_tF"(i1 %0) !dbg !35 {
  %2 = alloca i1, align 8
  %3 = bitcast i1* %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 1, i1 false)
  store i1 %0, i1* %2, align 8, !dbg !37
; CHECK:   B %bb.1, debug-location !{{[0-9]+}}
  br i1 %0, label %4, label %5, !dbg !38

4:                                                ; preds = %1
; Check that at -O0 the branches and their debug locations are not eliminated.
; CHECK:   B %bb.3, debug-location !{{[0-9]+}}
  br label %6, !dbg !39

5:                                                ; preds = %1
; CHECK:   B %bb.3, debug-location !{{[0-9]+}}
  br label %6, !dbg !40

6:                                                ; preds = %4, %5
  ret void, !dbg !39
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1
attributes #1 = { argmemonly nofree nosync nounwind willreturn writeonly }

!llvm.module.flags = !{!6, !7, !14}
!llvm.dbg.cu = !{!15, !27}

!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"Swift Version", i32 7}
!15 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !16, producer: "Swift", emissionKind: LineTablesOnly)
!16 = !DIFile(filename: "t.swift", directory: "/tmp")
!17 = !{}
!27 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !16, emissionKind: LineTablesOnly)
!35 = distinct !DISubprogram(name: "f", linkageName: "$s1t1f1bySb_tF", scope: !15, file: !16, line: 1, type: !36, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !17)
!36 = !DISubroutineType(types: null)
!37 = !DILocation(line: 0, scope: !35)
!38 = !DILocation(line: 2, column: 9, scope: !35)
!39 = !DILocation(line: 3, column: 1, scope: !35)
!40 = !DILocation(line: 2, column: 18, scope: !35)
