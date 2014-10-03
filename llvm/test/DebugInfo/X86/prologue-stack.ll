; RUN: llc -disable-fp-elim -O0 %s -mtriple x86_64-unknown-linux-gnu -o - | FileCheck %s

; int callme(int);
; int isel_line_test2() {
;   callme(400);
;   return 0;
; }

define i32 @isel_line_test2() nounwind uwtable {
  ; The stack adjustment should be part of the prologue.
  ; CHECK: isel_line_test2:
  ; CHECK: {{subq|leaq}} {{.*}}, %rsp
  ; CHECK: .loc 1 5 3 prologue_end
entry:
  %call = call i32 @callme(i32 400), !dbg !10
  ret i32 0, !dbg !12
}

declare i32 @callme(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.2 (trunk 164980) (llvm/trunk 164979)\000\00\000\00\000", metadata !13, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/bar.c] [DW_LANG_C99]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00isel_line_test2\00isel_line_test2\00\003\000\001\000\006\000\000\004", metadata !13, metadata !6, metadata !7, null, i32 ()* @isel_line_test2, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [isel_line_test2]
!6 = metadata !{metadata !"0x29", metadata !13} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 5, i32 3, metadata !11, null}
!11 = metadata !{metadata !"0xb\004\001\000", metadata !13, metadata !5} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/bar.c]
!12 = metadata !{i32 6, i32 3, metadata !11, null}
!13 = metadata !{metadata !"bar.c", metadata !"/usr/local/google/home/echristo/tmp"}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
