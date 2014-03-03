; RUN: opt < %s -add-discriminators -S | FileCheck %s

; Basic DWARF discriminator test. All the instructions in block
; 'if.then' should have a different discriminator value than
; the conditional branch at the end of block 'entry'.
;
; Original code:
;
;       void foo(int i) {
;         int x;
;         if (i < 10) x = i;
;       }

define void @foo(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32* %i.addr, align 4, !dbg !10
; CHECK:  %1 = load i32* %i.addr, align 4, !dbg !12

  store i32 %1, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %1, i32* %x, align 4, !dbg !12

  br label %if.end, !dbg !10
; CHECK:   br label %if.end, !dbg !12

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !12
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [basic.c] [DW_LANG_C99]
!1 = metadata !{metadata !"basic.c", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @foo, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [basic.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.5 "}
!10 = metadata !{i32 3, i32 0, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !4, i32 3, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [basic.c]
!12 = metadata !{i32 4, i32 0, metadata !4, null}

; CHECK: !12 = metadata !{i32 3, i32 0, metadata !13, null}
; CHECK: !13 = metadata !{i32 786443, metadata !1, metadata !11, i32 3, i32 0, i32 1, i32 0} ; [ DW_TAG_lexical_block ] [./basic.c]
; CHECK: !14 = metadata !{i32 4, i32 0, metadata !4, null}
