; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Check that we use DW_AT_low_pc

; CHECK: DW_TAG_compile_unit [1]
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_data4]
; CHECK: DW_TAG_subprogram [2]
; CHECK: DW_AT_low_pc [DW_FORM_addr]
; CHECK: DW_AT_high_pc [DW_FORM_data4]

; Function Attrs: nounwind uwtable
define void @z() #0 {
entry:
  ret void, !dbg !11
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/z.c] [DW_LANG_C99]
!1 = metadata !{metadata !"z.c", metadata !"/usr/local/google/home/echristo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"z", metadata !"z", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @z, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [z]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/z.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!11 = metadata !{i32 1, i32 0, metadata !4, null}
