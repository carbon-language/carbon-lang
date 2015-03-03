; RUN: llc -O0 < %s -o /dev/null
; llc should not crash on this invalid input.
; PR6588
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @foo() {
entry:
  call void @llvm.dbg.declare(metadata i32* undef, metadata !0, metadata !MDExpression())
  ret void
}

!0 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "sy", line: 890, arg: 0, scope: !1, file: !2, type: !7)
!1 = !MDSubprogram(name: "foo", linkageName: "foo", line: 892, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !8, scope: !3, type: !4)
!2 = !MDFile(filename: "qpainter.h", directory: "QtGui")
!3 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang 1.1", isOptimized: true, emissionKind: 0, file: !9, enums: !10, retainedTypes: !10)
!4 = !MDSubroutineType(types: !6)
!5 = !MDFile(filename: "splineeditor.cpp", directory: "src")
!6 = !{null}
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !MDFile(filename: "qpainter.h", directory: "QtGui")
!9 = !MDFile(filename: "splineeditor.cpp", directory: "src")
!10 = !{i32 0}
