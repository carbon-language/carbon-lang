; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj %s -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Ensure that pointer constants are emitted as unsigned data. Alternatively,
; these could be signless data (dataN).

; Built with Clang from:
; template <void *V, void (*F)(), int i>
; void func() {}
; template void func<nullptr, nullptr, 42>();

; CHECK: DW_TAG_subprogram
; CHECK:   DW_TAG_template_value_parameter
; CHECK:     DW_AT_name {{.*}} "V"
; CHECK:     DW_AT_const_value [DW_FORM_udata] (0)
; CHECK:   DW_TAG_template_value_parameter
; CHECK:     DW_AT_name {{.*}} "F"
; CHECK:     DW_AT_const_value [DW_FORM_udata] (0)

; Function Attrs: nounwind uwtable
define weak_odr void @_Z4funcILPv0ELPFvvE0ELi42EEvv() #0 {
entry:
  ret void, !dbg !18
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/constant-pointers.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"constant-pointers.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func<nullptr, nullptr, 42>", metadata !"func<nullptr, nullptr, 42>", metadata !"_Z4funcILPv0ELPFvvE0ELi42EEvv", i32 2, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z4funcILPv0ELPFvvE0ELi42EEvv, metadata !8, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [func<nullptr, nullptr, 42>]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/constant-pointers.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !9, metadata !11, metadata !13}
!9 = metadata !{i32 786480, null, metadata !"V", metadata !10, i8 0, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!11 = metadata !{i32 786480, null, metadata !"F", metadata !12, i8 0, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = metadata !{i32 786480, null, metadata !"i", metadata !14, i32 42, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!14 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!15 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!16 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!17 = metadata !{metadata !"clang version 3.5.0 "}
!18 = metadata !{i32 3, i32 0, metadata !4, null}
