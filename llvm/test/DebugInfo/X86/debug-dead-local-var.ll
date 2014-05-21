; RUN: llc -mtriple=x86_64-linux-gnu %s -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Reconstruct this via clang and -O2.
; static void foo() {
;   struct X { int a; int b; } xyz;
; }

; int bar() {
;   foo();
;   return 1;
; }

; Check that we still have the structure type for X even though we're not
; going to emit a low/high_pc for foo.
; CHECK: DW_TAG_structure_type

; Function Attrs: nounwind readnone uwtable
define i32 @bar() #0 {
entry:
  ret i32 1, !dbg !21
}

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 (trunk 209255) (llvm/trunk 209253)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/debug-dead-local-var.c] [DW_LANG_C99]
!1 = metadata !{metadata !"debug-dead-local-var.c", metadata !"/usr/local/google/home/echristo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !9}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"bar", metadata !"bar", metadata !"", i32 11, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 true, i32 ()* @bar, null, null, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [bar]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/debug-dead-local-var.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 6, metadata !10, i1 true, i1 true, i32 0, i32 0, null, i32 0, i1 true, null, null, null, metadata !12, i32 6} ; [ DW_TAG_subprogram ] [line 6] [local] [def] [foo]
!10 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{null}
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786688, metadata !9, metadata !"xyz", metadata !5, i32 8, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [xyz] [line 8]
!14 = metadata !{i32 786451, metadata !1, metadata !9, metadata !"X", i32 8, i64 64, i64 32, i32 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [X] [line 8, size 64, align 32, offset 0] [def] [from ]
!15 = metadata !{metadata !16, metadata !17}
!16 = metadata !{i32 786445, metadata !1, metadata !14, metadata !"a", i32 8, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_member ] [a] [line 8, size 32, align 32, offset 0] [from int]
!17 = metadata !{i32 786445, metadata !1, metadata !14, metadata !"b", i32 8, i64 32, i64 32, i64 32, i32 0, metadata !8} ; [ DW_TAG_member ] [b] [line 8, size 32, align 32, offset 32] [from int]
!18 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!19 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!20 = metadata !{metadata !"clang version 3.5.0 (trunk 209255) (llvm/trunk 209253)"}
!21 = metadata !{i32 13, i32 0, metadata !4, null}
