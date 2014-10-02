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

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/constant-pointers.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"constant-pointers.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00func<nullptr, nullptr, 42>\00func<nullptr, nullptr, 42>\00_Z4funcILPv0ELPFvvE0ELi42EEvv\002\000\001\000\006\00256\000\002", metadata !1, metadata !5, metadata !6, null, void ()* @_Z4funcILPv0ELPFvvE0ELi42EEvv, metadata !8, null, metadata !2} ; [ DW_TAG_subprogram ] [line 2] [def] [func<nullptr, nullptr, 42>]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/constant-pointers.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !9, metadata !11, metadata !13}
!9 = metadata !{metadata !"0x30\00V\000\000", null, metadata !10, i8 0, null} ; [ DW_TAG_template_value_parameter ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!11 = metadata !{metadata !"0x30\00F\000\000", null, metadata !12, i8 0, null} ; [ DW_TAG_template_value_parameter ]
!12 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = metadata !{metadata !"0x30\00i\000\000", null, metadata !14, i32 42, null} ; [ DW_TAG_template_value_parameter ]
!14 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!15 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!16 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!17 = metadata !{metadata !"clang version 3.5.0 "}
!18 = metadata !{i32 3, i32 0, metadata !4, null}
