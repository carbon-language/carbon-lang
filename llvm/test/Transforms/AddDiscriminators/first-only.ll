; RUN: opt < %s -add-discriminators -S | FileCheck %s

; Test that the only instructions that receive a new discriminator in
; the block 'if.then' are those that share the same line number as
; the branch in 'entry'.
;
; Original code:
;
;       void foo(int i) {
;         int x, y;
;         if (i < 10) { x = i;
;             y = -i;
;         }
;       }

define void @foo(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32* %i.addr, align 4, !dbg !12
  store i32 %1, i32* %x, align 4, !dbg !12

  %2 = load i32* %i.addr, align 4, !dbg !14
; CHECK:  %2 = load i32* %i.addr, align 4, !dbg !15

  %sub = sub nsw i32 0, %2, !dbg !14
; CHECK:  %sub = sub nsw i32 0, %2, !dbg !15

  store i32 %sub, i32* %y, align 4, !dbg !14
; CHECK:  store i32 %sub, i32* %y, align 4, !dbg !15

  br label %if.end, !dbg !15
; CHECK:  br label %if.end, !dbg !16

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !16
; CHECK:  ret void, !dbg !17
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5 (trunk 199750) (llvm/trunk 199751)\000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [first-only.c] [DW_LANG_C99]
!1 = metadata !{metadata !"first-only.c", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, void (i32)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [first-only.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!9 = metadata !{metadata !"clang version 3.5 (trunk 199750) (llvm/trunk 199751)"}
!10 = metadata !{i32 3, i32 0, metadata !11, null}

!11 = metadata !{metadata !"0xb\003\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [first-only.c]
; CHECK: !11 = metadata !{metadata !"0xb\003\000\000", metadata !1, metadata !4}

!12 = metadata !{i32 3, i32 0, metadata !13, null}

!13 = metadata !{metadata !"0xb\003\000\001", metadata !1, metadata !11} ; [ DW_TAG_lexical_block ] [first-only.c]
; CHECK: !13 = metadata !{metadata !"0xb\001", metadata !1, metadata !14} ; [ DW_TAG_lexical_block ] [./first-only.c]

!14 = metadata !{i32 4, i32 0, metadata !13, null}
; CHECK: !14 = metadata !{metadata !"0xb\003\000\001", metadata !1, metadata !11}

!15 = metadata !{i32 5, i32 0, metadata !13, null}
; CHECK: !15 = metadata !{i32 4, i32 0, metadata !14, null}

!16 = metadata !{i32 6, i32 0, metadata !4, null}
; CHECK: !16 = metadata !{i32 5, i32 0, metadata !14, null}
; CHECK: !17 = metadata !{i32 6, i32 0, metadata !4, null}

