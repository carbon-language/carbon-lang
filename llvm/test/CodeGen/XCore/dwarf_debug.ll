; RUN: llc < %s -mtriple=xcore-unknown-unknown -O0 | FileCheck %s

; target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32-f64:32-a:0:32-n32"
; target triple = "xcore"

; CHECK-LABEL: f
; CHECK: entsp 2
; ...the prologue...
; CHECK: .loc 1 2 0 prologue_end      # :2:0
; CHECK: add r0, r0, 1
; CHECK: retsp 2
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !11, metadata !MDExpression()), !dbg !12
  %0 = load i32, i32* %a.addr, align 4, !dbg !12
  %add = add nsw i32 %0, 1, !dbg !12
  ret i32 %add, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!0 = !MDCompileUnit(language: DW_LANG_C99, isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !MDFile(filename: "", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !MDSubprogram(name: "f", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !1, scope: !5, type: !6, function: i32 (i32)* @f, variables: !2)
!5 = !MDFile(filename: "", directory: "")
!6 = !MDSubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 2, arg: 1, scope: !4, file: !5, type: !8)
!12 = !MDLocation(line: 2, scope: !4)

