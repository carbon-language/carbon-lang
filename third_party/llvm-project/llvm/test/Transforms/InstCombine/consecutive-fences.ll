; RUN: opt -instcombine -S %s | FileCheck %s

; Make sure we collapse the fences in this case

; CHECK-LABEL: define void @tinkywinky
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   fence syncscope("singlethread") acquire
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @tinkywinky() {
  fence seq_cst
  fence seq_cst
  fence seq_cst
  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") acquire
  ret void
}

; CHECK-LABEL: define void @dipsy
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @dipsy() {
  fence seq_cst
  fence syncscope("singlethread") seq_cst
  ret void
}

; CHECK-LABEL: define void @patatino
; CHECK-NEXT:   fence acquire
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   fence acquire
; CHECK-NEXT:   fence seq_cst
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define void @patatino() {
  fence acquire
  fence seq_cst
  fence acquire
  fence seq_cst
  ret void
}

; CHECK-LABEL: define void @debug
; CHECK-NOT: fence
; CHECK: call void @llvm.dbg.value
; CHECK: fence seq_cst
define void @debug() {
  fence seq_cst
  tail call void @llvm.dbg.value(metadata i32 5, metadata !1, metadata !DIExpression()), !dbg !9
  fence seq_cst
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "Me", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: null, retainedTypes: null, imports: null)
!1 = !DILocalVariable(name: "", arg: 1, scope: !2, file: null, line: 1, type: null)
!2 = distinct !DISubprogram(name: "debug", linkageName: "debug", scope: null, file: null, line: 0, type: null, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!3 = !DIFile(filename: "consecutive-fences.ll", directory: "")
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 7, !"PIC Level", i32 2}
!9 = !DILocation(line: 0, column: 0, scope: !2)
