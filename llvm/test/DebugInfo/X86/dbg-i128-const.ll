; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s

; CHECK: DW_AT_const_value
; CHECK-NEXT: 42

define i128 @__foo(i128 %a, i128 %b) nounwind {
entry:
  tail call void @llvm.dbg.value(metadata i128 42 , i64 0, metadata !1, metadata !MDExpression()), !dbg !11
  %add = add i128 %a, %b, !dbg !11
  ret i128 %add, !dbg !11
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!16}

!0 = !{i128 42 }
!1 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "MAX", line: 29, scope: !2, file: !4, type: !8)
!2 = distinct !MDLexicalBlock(line: 26, column: 0, file: !13, scope: !3)
!3 = !MDSubprogram(name: "__foo", linkageName: "__foo", line: 26, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 26, file: !13, scope: !4, type: !6, function: i128 (i128, i128)* @__foo)
!4 = !MDFile(filename: "foo.c", directory: "/tmp")
!5 = !MDCompileUnit(language: DW_LANG_C89, producer: "clang", isOptimized: true, emissionKind: 0, file: !13, enums: !15, retainedTypes: !15, subprograms: !12, imports:  null)
!6 = !MDSubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !MDDerivedType(tag: DW_TAG_typedef, name: "ti_int", line: 78, file: !14, scope: !4, baseType: !10)
!9 = !MDFile(filename: "myint.h", directory: "/tmp")
!10 = !MDBasicType(tag: DW_TAG_base_type, size: 128, align: 128, encoding: DW_ATE_signed)
!11 = !MDLocation(line: 29, scope: !2)
!12 = !{!3}
!13 = !MDFile(filename: "foo.c", directory: "/tmp")
!14 = !MDFile(filename: "myint.h", directory: "/tmp")
!15 = !{i32 0}
!16 = !{i32 1, !"Debug Info Version", i32 3}
