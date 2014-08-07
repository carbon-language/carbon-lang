; REQUIRES: object-emission
; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s

; Generated from the following source compiled with clang -gmlt:
; void f1(void) {}
; void f2(void) __attribute__((section("__TEXT,__bar"))) {}

; Check that
;  * -gmlt ('Emission Kind' of 'LineTablesOnly' in the CU debug info metadata)
;    doesn't produce ranges.
;  * if no ranges are produced, no debug_ranges list (not even an empty one) is
;    emitted.

; -gmlt means no DW_AT_ranges on the CU, even though there are parts of the CU
; in different sections and this would normally necessitate a DW_AT_ranges
; attribute on the CU.
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_AT_ranges
; CHECK: {{DW_TAG|NULL}}

; FIXME: We probably want to avoid printing out anything if the section isn't there.
; CHECK: .debug_ranges contents:
; CHECK-NOT: 00000000 <End of list>

; Check that we don't emit any pubnames or pubtypes under -gmlt
; CHECK: .debug_pubnames contents:
; CHECK-NOT: Offset

; CHECK: .debug_pubtypes contents:
; CHECK-NOT: Offset

; Function Attrs: nounwind uwtable
define void @f1() #0 {
entry:
  ret void, !dbg !11
}

; Function Attrs: nounwind uwtable
define void @f2() #0 section "__TEXT,__bar" {
entry:
  ret void, !dbg !12
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/cu-line-tables.c] [DW_LANG_C99]
!1 = metadata !{metadata !"cu-line-tables.c", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !7}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f1", metadata !"f1", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @f1, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [f1]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/cu-line-tables.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f2", metadata !"f2", metadata !"", i32 2, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @f2, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [f2]
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.6.0 "}
!11 = metadata !{i32 1, i32 16, metadata !4, null}
!12 = metadata !{i32 2, i32 48, metadata !7, null}
