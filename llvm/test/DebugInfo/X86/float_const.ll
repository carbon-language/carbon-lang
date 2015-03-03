; RUN: llc < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; from (at -Os):
; void foo() {
;   float a = 3.14;
;   *(int *)&a = 0;
; }
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind optsize readnone uwtable
define void @foo() #0 {
entry:
  tail call void @llvm.dbg.declare(metadata float* undef, metadata !13, metadata !19), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 1078523331, i64 0, metadata !13, metadata !19), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !13, metadata !19), !dbg !20
; CHECK:  DW_AT_const_value [DW_FORM_sdata]    (0)
; CHECK-NEXT: DW_AT_name {{.*}}"a"
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind optsize readnone uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.7.0 (trunk 227686)", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !3, subprograms: !6, globals: !2, imports: !2)
!1 = !MDFile(filename: "foo.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !5)
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{!7}
!7 = !MDSubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, isOptimized: true, scopeLine: 1, file: !8, scope: !9, type: !10, function: void ()* @foo, variables: !12)
!8 = !MDFile(filename: "foo.c", directory: "")
!9 = !MDFile(filename: "foo.c", directory: "")
!10 = !MDSubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "a", line: 2, scope: !7, file: !9, type: !14)
!14 = !MDBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!15 = !{i32 2, !"Dwarf Version", i32 2}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"PIC Level", i32 2}
!18 = !{!"clang version 3.7.0 (trunk 227686)"}
!19 = !MDExpression()
!20 = !MDLocation(line: 2, column: 9, scope: !7)
!21 = !MDLocation(line: 4, column: 1, scope: !7)
