; RUN: llc -mtriple=i386-unknown-unknown -mcpu=core2 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s
;
; Generated from:
;
;   int foo(int i) {
;     if (i < 10) return i - 1;
;     return 0;
;   }
;
; Manually generated debug nodes !14 and !15 to incorporate an
; arbitrary discriminator with value 42.

define i32 @foo(i32 %i) #0 {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32* %i.addr, align 4, !dbg !14
  %sub = sub nsw i32 %1, 1, !dbg !14
  store i32 %sub, i32* %retval, !dbg !14
  br label %return, !dbg !14

if.end:                                           ; preds = %entry
  store i32 0, i32* %retval, !dbg !12
  br label %return, !dbg !12

return:                                           ; preds = %if.end, %if.then
  %2 = load i32* %retval, !dbg !13
  ret i32 %2, !dbg !13
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [./discriminator.c] [DW_LANG_C99]
!1 = metadata !{metadata !"discriminator.c", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, i32 (i32)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [./discriminator.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!9 = metadata !{metadata !"clang version 3.5 "}
!10 = metadata !{i32 2, i32 0, metadata !11, null}
!11 = metadata !{metadata !"0xb\002\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [./discriminator.c]
!12 = metadata !{i32 3, i32 0, metadata !4, null}
!13 = metadata !{i32 4, i32 0, metadata !4, null}
!14 = metadata !{i32 2, i32 0, metadata !15, null}
!15 = metadata !{metadata !"0xb\0042", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [./discriminator.c]

; CHECK: Address            Line   Column File   ISA Discriminator Flags
; CHECK: ------------------ ------ ------ ------ --- ------------- -------------
; CHECK: 0x0000000000000011      2      0      1   0            42  is_stmt
