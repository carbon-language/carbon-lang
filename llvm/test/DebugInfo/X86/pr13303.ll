; RUN: llc %s -o %t -filetype=obj -mtriple=x86_64-unknown-linux-gnu
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s
; PR13303

; Check that the prologue ends with is_stmt here.
; CHECK: 0x0000000000000000 {{.*}} is_stmt

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.2 (trunk 160143)\000\00\000\00\000", metadata !12, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ] [/home/probinson/PR13303.c] [DW_LANG_C99]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00main\00main\00\001\000\001\000\006\000\000\001", metadata !12, metadata !6, metadata !7, null, i32 ()* @main, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 1] [def] [main]
!6 = metadata !{metadata !"0x29", metadata !12} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 1, i32 14, metadata !11, null}
!11 = metadata !{metadata !"0xb\001\0012\000", metadata !12, metadata !5} ; [ DW_TAG_lexical_block ] [/home/probinson/PR13303.c]
!12 = metadata !{metadata !"PR13303.c", metadata !"/home/probinson"}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
