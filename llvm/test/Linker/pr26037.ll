; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/pr26037.ll -o %t2.bc
; RUN: llvm-link -S -only-needed %t2.bc %t.bc | FileCheck %s

; CHECK: distinct !DISubprogram(name: "a"
; CHECK: !DIImportedEntity({{.*}}, entity:

define void @_ZN1A1aEv() #0 !dbg !4 {
entry:
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 256934) (llvm/trunk 256936)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3, imports: !8)
!1 = !DIFile(filename: "a2.cc", directory: "")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "a", linkageName: "_ZN1A1aEv", scope: !5, file: !1, line: 5, type: !6, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DINamespace(name: "A", scope: null, file: !1, line: 1)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9}
!9 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !4, line: 2)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.8.0 (trunk 256934) (llvm/trunk 256936)"}
!13 = !DILocation(line: 6, column: 1, scope: !4)
