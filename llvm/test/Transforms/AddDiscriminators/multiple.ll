; RUN: opt < %s -add-discriminators -S | FileCheck %s

; Discriminator support for multiple CFG paths on the same line.
;
;       void foo(int i) {
;         int x;
;         if (i < 10) x = i; else x = -i;
;       }
;
; The two stores inside the if-then-else line must have different discriminator
; values.

define void @foo(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.else, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32* %i.addr, align 4, !dbg !10
; CHECK:  %1 = load i32* %i.addr, align 4, !dbg !12

  store i32 %1, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %1, i32* %x, align 4, !dbg !12

  br label %if.end, !dbg !10
; CHECK:  br label %if.end, !dbg !12

if.else:                                          ; preds = %entry
  %2 = load i32* %i.addr, align 4, !dbg !10
; CHECK:  %2 = load i32* %i.addr, align 4, !dbg !14

  %sub = sub nsw i32 0, %2, !dbg !10
; CHECK:  %sub = sub nsw i32 0, %2, !dbg !14

  store i32 %sub, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %sub, i32* %x, align 4, !dbg !14

  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !12
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 (trunk 199750) (llvm/trunk 199751)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [multiple.c] [DW_LANG_C99]
!1 = metadata !{metadata !"multiple.c", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @foo, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [multiple.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.5 (trunk 199750) (llvm/trunk 199751)"}
!10 = metadata !{i32 3, i32 0, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !4, i32 3, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [multiple.c]
!12 = metadata !{i32 4, i32 0, metadata !4, null}

; CHECK: !12 = metadata !{i32 3, i32 0, metadata !13, null}
; CHECK: !13 = metadata !{i32 786443, metadata !1, metadata !11, i32 3, i32 0, i32 1, i32 0} ; [ DW_TAG_lexical_block ] [./multiple.c]
; CHECK: !14 = metadata !{i32 3, i32 0, metadata !15, null}
; CHECK: !15 = metadata !{i32 786443, metadata !1, metadata !11, i32 3, i32 0, i32 2, i32 1} ; [ DW_TAG_lexical_block ] [./multiple.c]
