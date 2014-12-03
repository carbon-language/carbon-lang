; Inject metadata to set the .gcno file location
; RUN: echo '!19 = metadata !{metadata !"%/T/return-block.ll", metadata !0}' > %t1
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
  %0 = load i32* @A, align 4, !dbg !15
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

!0 = metadata !{metadata !"0x11\0012\00clang version 3.6.0 (trunk 223182)\001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !8, metadata !2} ; [ DW_TAG_compile_unit ] [return-block.c] [DW_LANG_C99]
!1 = metadata !{metadata !".../llvm/test/Transforms/GCOVProfiling/return-block.ll", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00test\00test\00\005\000\001\000\000\000\001\005", metadata !1, metadata !5, metadata !6, null, void ()* @test, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 5] [def] [test]
!5 = metadata !{metadata !"0x29", metadata !1}    ; [ DW_TAG_file_type ] [return-block.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x34\00A\00A\00\003\000\001", null, metadata !5, metadata !10, i32* @A, null} ; [ DW_TAG_variable ] [A] [line 3] [def]
!10 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!12 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!13 = metadata !{metadata !"clang version 3.6.0 (trunk 223182)"}
!14 = metadata !{i32 6, i32 3, metadata !4, null}
!15 = metadata !{i32 7, i32 7, metadata !4, null}
!16 = metadata !{i32 8, i32 5, metadata !17, null}
!17 = metadata !{metadata !"0xb\007\007\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [return-block.c]
!18 = metadata !{i32 9, i32 1, metadata !4, null}

; There should be no destination edges for block 1.
; CHECK: Block : 0 Counter : 0
; CHECK-NEXT:         Destination Edges : 2 (0), 
; CHECK-NEXT: Block : 1 Counter : 0
; CHECK-NEXT:         Source Edges : 4 (0), 
; CHECK-NEXT: Block : 2 Counter : 0
