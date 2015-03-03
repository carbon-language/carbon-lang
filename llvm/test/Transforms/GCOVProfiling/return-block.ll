; Inject metadata to set the .gcno file location
; RUN: echo '!19 = !{!"%/T/return-block.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output %t2
; RUN: llvm-cov gcov -n -dump %T/return-block.gcno 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @test() #0 {
entry:
  tail call void (...)* @f() #2, !dbg !14
  %0 = load i32, i32* @A, align 4, !dbg !15
  %tobool = icmp eq i32 %0, 0, !dbg !15
  br i1 %tobool, label %if.end, label %if.then, !dbg !15

if.then:                                          ; preds = %entry
  tail call void (...)* @g() #2, !dbg !16
  br label %if.end, !dbg !16

if.end:                                           ; preds = %entry, %if.then
  ret void, !dbg !18
}

declare void @f(...) #1

declare void @g(...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.gcov = !{!19}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 (trunk 223182)", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !8, imports: !2)
!1 = !MDFile(filename: ".../llvm/test/Transforms/GCOVProfiling/return-block.ll", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !MDSubprogram(name: "test", line: 5, isLocal: false, isDefinition: true, isOptimized: true, scopeLine: 5, file: !1, scope: !5, type: !6, function: void ()* @test, variables: !2)
!5 = !MDFile(filename: ".../llvm/test/Transforms/GCOVProfiling/return-block.ll", directory: "")
!6 = !MDSubroutineType(types: !7)
!7 = !{null}
!8 = !{!9}
!9 = !MDGlobalVariable(name: "A", line: 3, isLocal: false, isDefinition: true, scope: null, file: !5, type: !10, variable: i32* @A)
!10 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.6.0 (trunk 223182)"}
!14 = !MDLocation(line: 6, column: 3, scope: !4)
!15 = !MDLocation(line: 7, column: 7, scope: !4)
!16 = !MDLocation(line: 8, column: 5, scope: !17)
!17 = distinct !MDLexicalBlock(line: 7, column: 7, file: !1, scope: !4)
!18 = !MDLocation(line: 9, column: 1, scope: !4)

; There should be no destination edges for block 1.
; CHECK: Block : 0 Counter : 0
; CHECK-NEXT:         Destination Edges : 2 (0), 
; CHECK-NEXT: Block : 1 Counter : 0
; CHECK-NEXT:         Source Edges : 4 (0), 
; CHECK-NEXT: Block : 2 Counter : 0
