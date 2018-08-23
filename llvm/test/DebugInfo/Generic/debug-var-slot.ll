; After adding new debug instruction for labels, it is possible to have
; debug instructions before DBG_VALUE. When querying DBG_VALUE's slot
; index using previous instruction and the previous instruction is debug
; instruction, it will trigger an assertion as using debug instruction
; to get slot index. This test is to emulate the case when DBG_VALUE's
; previous instruction is DBG_LABEL in LiveDebugVariables pass.
;
; RUN: llc < %s -stop-after=livedebugvars -debug 2>&1 | FileCheck %s
;
; CHECK: COMPUTING LIVE DEBUG VARIABLES: foo
; CHECK: DEBUG VARIABLES
; CHECK-NEXT: "local_var,7"

source_filename = "debug-var-slot.c"

define dso_local i32 @foo(i32 %a, i32 %b) !dbg !6 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  br label %top

top:
  call void @llvm.dbg.label(metadata !10), !dbg !13
  call void @llvm.dbg.value(metadata i32 %0, metadata !12, metadata !DIExpression()), !dbg !14
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %a.addr, align 4
  %2 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %1, %2
  store i32 %add, i32* %sum, align 4
  br label %done

done:
  %3 = load i32, i32* %sum, align 4
  ret i32 %3, !dbg !15
}

declare void @llvm.dbg.label(metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "debug-var-slot.c", directory: "./")
!2 = !{}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILabel(scope: !6, name: "top", file: !1, line: 4)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "local_var", scope: !6, file: !1, line: 7, type: !11)
!13 = !DILocation(line: 4, column: 1, scope: !6)
!14 = !DILocation(line: 7, column: 1, scope: !6)
!15 = !DILocation(line: 8, column: 3, scope: !6)
