; RUN: llc -dwarf-fission=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

@a = common global i32 0, align 4

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 12, metadata !"baz.c", metadata !"/usr/local/google/home/echristo/tmp", metadata !"clang version 3.3 (trunk 169021) (llvm/trunk 169020)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/baz.c] [DW_LANG_C99]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, i32* @a} ; [ DW_TAG_variable ] [a] [line 1] [def]
!6 = metadata !{i32 786473, metadata !"baz.c", metadata !"/usr/local/google/home/echristo/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]

; Check that the skeleton compile unit contains the proper attributes:
; This DIE has the following attributes: DW_AT_comp_dir, DW_AT_stmt_list,
; DW_AT_low_pc, DW_AT_high_pc, DW_AT_ranges, DW_AT_dwo_name, DW_AT_dwo_id,
; DW_AT_ranges_base, DW_AT_addr_base.

; CHECK: DW_TAG_compile_unit [1]  
; CHECK: DW_AT_GNU_dwo_name [DW_FORM_strp] ( .debug_str[0x00000035] = "baz.c")
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK: DW_AT_stmt_list [DW_FORM_data4]   (0x00000000)
; CHECK: DW_AT_comp_dir [DW_FORM_strp]     ( .debug_str[0x0000003b] = "/usr/local/google/home/echristo/tmp")
