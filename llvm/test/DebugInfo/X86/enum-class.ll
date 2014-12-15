; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

@a = global i32 0, align 4
@b = global i64 0, align 8
@c = global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}

!0 = !{!"0x11\004\00clang version 3.2 (trunk 157269) (llvm/trunk 157264)\000\00\000\00\000", !22, !1, !15, !15, !17,  !15} ; [ DW_TAG_compile_unit ]
!1 = !{!3, !8, !12}
!3 = !{!"0x4\00A\001\0032\0032\000\000\000", !4, null, !5, !6, null, null, null} ; [ DW_TAG_enumeration_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from int]
!4 = !{!"0x29", !22} ; [ DW_TAG_file_type ]
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!6 = !{!7}
!7 = !{!"0x28\00A1\001"} ; [ DW_TAG_enumerator ]
!8 = !{!"0x4\00B\002\0064\0064\000\000\000", !4, null, !9, !10, null, null, null} ; [ DW_TAG_enumeration_type ] [B] [line 2, size 64, align 64, offset 0] [def] [from long unsigned int]
!9 = !{!"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ]
!10 = !{!11}
!11 = !{!"0x28\00B1\001"} ; [ DW_TAG_enumerator ]
!12 = !{!"0x4\00C\003\0032\0032\000\000\000", !4, null, null, !13, null, null, null} ; [ DW_TAG_enumeration_type ] [C] [line 3, size 32, align 32, offset 0] [def] [from ]
!13 = !{!14}
!14 = !{!"0x28\00C1\001"} ; [ DW_TAG_enumerator ]
!15 = !{}
!17 = !{!19, !20, !21}
!19 = !{!"0x34\00a\00a\00\004\000\001", null, !4, !3, i32* @a, null} ; [ DW_TAG_variable ]
!20 = !{!"0x34\00b\00b\00\005\000\001", null, !4, !8, i64* @b, null} ; [ DW_TAG_variable ]
!21 = !{!"0x34\00c\00c\00\006\000\001", null, !4, !12, i32* @c, null} ; [ DW_TAG_variable ]
!22 = !{!"foo.cpp", !"/Users/echristo/tmp"}

; CHECK: DW_TAG_enumeration_type [{{.*}}]
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "A")

; CHECK: DW_TAG_enumeration_type [{{.*}}] *
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]          ( .debug_str[{{.*}}] = "B")

; CHECK: DW_TAG_enumeration_type [6]
; CHECK-NOT: DW_AT_enum_class
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "C")
!23 = !{i32 1, !"Debug Info Version", i32 2}
