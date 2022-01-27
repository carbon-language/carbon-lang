; RUN: llc -start-before=codegenprepare -stop-after=codegenprepare -mtriple=x86_64-unknown-unknown %s -o - | FileCheck %s
;
; CGP duplicates address calculation into each basic block that contains loads
; or stores, so that they can be folded into instruction memory operands for
; example. dbg.value's should be redirected to identify such local address
; computations, to give the best opportunity for variable locations to be
; preserved.
; This test has two dbg.values in it, one before and one after the relevant
; memory instruction. Test that the one before does _not_ get updated (as that
; would either make it use-before-def or shift when the variable appears), and
; that the dbg.value after the memory instruction does get updated.

define dso_local i8 @foo(i32 *%p, i32 %cond) !dbg !7 {
entry:
; There should be no dbg.values in this block.
; CHECK-LABEL: entry:
; CHECK-NOT:   dbg.value
  %casted = bitcast i32 *%p to i8*
  %arith = getelementptr i8, i8 *%casted, i32 3
  %load1 = load i8, i8 *%arith
  %cmpresult = icmp eq i32 %cond, 0
  br i1 %cmpresult, label %next, label %ret

next:
; Address calcs should be duplicated into this block. One dbg.value should be
; updated, and the other should not.
; CHECK-LABEL: next:
; CHECK:       dbg.value(metadata i8* %arith, metadata ![[DIVAR:[0-9]+]],
; CHECK-SAME:    metadata !DIExpression()
; CHECK-NEXT:  %[[CASTVAR:[0-9a-zA-Z]+]] = bitcast i32* %p to i8*
; CHECK-NEXT:  %[[GEPVAR:[0-9a-zA-Z]+]] = getelementptr i8, i8* %[[CASTVAR]],
; CHECK-SAME:                             i64 3
; CHECK-NEXT:  %loaded = load i8, i8* %[[GEPVAR]]
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i8* %[[GEPVAR]],
; CHECK-SAME:                            metadata ![[DIVAR]],
; CHECK-NEXT:  call void @llvm.dbg.value(metadata !DIArgList(i8* %[[GEPVAR]],
; CHECK-SAME:                            i8* %[[GEPVAR]]), metadata ![[DIVAR]],
  call void @llvm.dbg.value(metadata i8 *%arith, metadata !12, metadata !DIExpression()), !dbg !14
  %loaded = load i8, i8 *%arith
  call void @llvm.dbg.value(metadata i8 *%arith, metadata !12, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata !DIArgList(i8 *%arith, i8 *%arith), metadata !12, metadata !DIExpression()), !dbg !14
  ret i8 %loaded

ret:
  ret i8 0
}

; CHECK: ![[DIVAR]] = !DILocalVariable(name: "p",

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 348209)"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 4, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 4, type: !10)
!14 = !DILocation(line: 4, column: 15, scope: !7)
!20 = distinct !DILexicalBlock(scope: !7, file: !1, line: 8, column: 7)
