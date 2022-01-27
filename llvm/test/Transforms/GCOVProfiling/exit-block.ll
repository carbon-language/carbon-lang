; Inject metadata to set the .gcno file location
; RUN: rm -rf %t && mkdir -p %t
; RUN: echo '!19 = !{!"%/t/exit-block.ll", !0}' > %t/1
; RUN: cat %s %t/1 > %t/2

; By default, the exit block is the second.
; RUN: opt -insert-gcov-profiling -disable-output %t/2
; RUN: llvm-cov gcov -n -dump %t/exit-block.gcno 2>&1 | FileCheck --check-prefixes=CHECK,EXIT-SECOND %s

; But we can optionally emit it last, to match GCC<4.8 (r189778).
; RUN: opt -insert-gcov-profiling -default-gcov-version='407*' -disable-output %t/2
; RUN: llvm-cov gcov -n -dump %t/exit-block.gcno 2>&1 | FileCheck --check-prefixes=CHECK,EXIT-LAST %s
; RUN: rm  %t/exit-block.gcno

; By default, the exit block is the second.
; RUN: opt -passes=insert-gcov-profiling -disable-output %t/2
; RUN: llvm-cov gcov -n -dump %t/exit-block.gcno 2>&1 | FileCheck --check-prefixes=CHECK,EXIT-SECOND %s

; But we can optionally emit it last, to match GCC<4.8 (r189778).
; RUN: opt -passes=insert-gcov-profiling -default-gcov-version='407*' -disable-output %t/2
; RUN: llvm-cov gcov -n -dump %t/exit-block.gcno 2>&1 | FileCheck --check-prefixes=CHECK,EXIT-LAST %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global i32 0, align 4, !dbg !9

; Function Attrs: nounwind uwtable
define void @test() #0 !dbg !4 {
entry:
  tail call void (...) @f() #2, !dbg !14
  %0 = load i32, i32* @A, align 4, !dbg !15
  %tobool = icmp eq i32 %0, 0, !dbg !15
  br i1 %tobool, label %if.end, label %if.then, !dbg !15

if.then:                                          ; preds = %entry
  tail call void (...) @g() #2, !dbg !16
  br label %if.end, !dbg !16

if.end:                                           ; preds = %entry, %if.then
  ret void, !dbg !18
}

declare void @f(...) #1

declare void @g(...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.gcov = !{!19}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 (trunk 223182)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !8, imports: !2)
!1 = !DIFile(filename: ".../llvm/test/Transforms/GCOVProfiling/exit-block.ll", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "test", line: 5, isLocal: false, isDefinition: true, isOptimized: true, unit: !0, scopeLine: 5, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: ".../llvm/test/Transforms/GCOVProfiling/exit-block.ll", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9}
!9 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "A", line: 3, isLocal: false, isDefinition: true, scope: null, file: !5, type: !10), expr: !DIExpression())
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.6.0 (trunk 223182)"}
!14 = !DILocation(line: 6, column: 3, scope: !4)
!15 = !DILocation(line: 7, column: 7, scope: !4)
!16 = !DILocation(line: 8, column: 5, scope: !17)
!17 = distinct !DILexicalBlock(line: 7, column: 7, file: !1, scope: !4)
!18 = !DILocation(line: 9, column: 1, scope: !4)

; There should be no destination edges for the exit block.
; CHECK: Block : 1 Counter : 0
; EXIT-LAST:       Destination Edges
; EXIT-SECOND-NOT: Destination Edges
; CHECK: Block : 2 Counter : 0
; CHECK: Block : 4 Counter : 0
; EXIT-LAST-NOT: Destination Edges
; EXIT-SECOND:   Destination Edges
; CHECK-NOT: Block :
