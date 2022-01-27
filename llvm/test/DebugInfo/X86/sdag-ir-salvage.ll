; RUN: llc -mtriple=x86_64-unknown-unknown -start-after=codegenprepare \
; RUN:    -stop-before finalize-isel %s -o -  \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE
; RUN: llc -mtriple=x86_64-unknown-unknown -start-after=codegenprepare \
; RUN:    -stop-before finalize-isel %s -o -  \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF

; Test that the dbg.value for %baz, which doesn't exist in the 'next' bb,
; can be salvaged back to the underlying argument vreg.

; CHECK:       ![[AAAVAR:.*]] = !DILocalVariable(name: "aaa",
; CHECK-LABEL: bb.0.entry:
; INSTRREF:    DBG_PHI $rdi, 1
; CHECK-LABEL: bb.1.next:
; INSTRREF:    DBG_INSTR_REF 1, 0, ![[AAAVAR]]
; DBGVALUE:    DBG_VALUE %{{[0-9]+}}, $noreg, ![[AAAVAR]]

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-linux-gnu"

define i8 @f(i32* %foo) local_unnamed_addr !dbg !6 {
entry:
  %bar = getelementptr i32, i32* %foo, i32 4
  %baz = bitcast i32* %bar to i8*
  %quux = load i8, i8* %baz
  br label %next

next:                                             ; preds = %entry
  tail call void @llvm.dbg.value(metadata i8* %baz, metadata !15, metadata !DIExpression()), !dbg !30
  %xyzzy = add i8 %quux, 123
  br label %fin

fin:                                              ; preds = %next
  %trains = getelementptr i32, i32* %foo, i32 3
  %planes = bitcast i32* %trains to i8*
  %cars = load i8, i8* %planes
  %ret = add i8 %xyzzy, %cars
  ret i8 %ret
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26, !27, !28}
!llvm.ident = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: ".")
!2 = !{}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 18, type: !7, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!7 = !DISubroutineType(types: !8)
!8 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
!14 = !{!15}
!15 = !DILocalVariable(name: "aaa", scope: !6, file: !1, line: 18, type: !13)
!25 = !{i32 2, !"Dwarf Version", i32 4}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"wchar_size", i32 4}
!28 = !{i32 7, !"PIC Level", i32 2}
!29 = !{!"clang"}
!30 = !DILocation(line: 18, column: 14, scope: !6)
