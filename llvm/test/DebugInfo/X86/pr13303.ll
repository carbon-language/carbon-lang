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

!0 = metadata !{i32 786449, i32 12, metadata !6, metadata !"clang version 3.2 (trunk 160143)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ] [/home/probinson/PR13303.c] [DW_LANG_C99]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !6, metadata !"main", metadata !"main", metadata !"", metadata !6, i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @main, null, null, metadata !1, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [main]
!6 = metadata !{i32 786473, metadata !"PR13303.c", metadata !"/home/probinson", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 1, i32 14, metadata !11, null}
!11 = metadata !{i32 786443, metadata !6, metadata !5, i32 1, i32 12, i32 0} ; [ DW_TAG_lexical_block ] [/home/probinson/PR13303.c]
