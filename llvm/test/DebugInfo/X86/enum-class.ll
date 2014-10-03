; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

@a = global i32 0, align 4
@b = global i64 0, align 8
@c = global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}

!0 = metadata !{metadata !"0x11\004\00clang version 3.2 (trunk 157269) (llvm/trunk 157264)\000\00\000\00\000", metadata !22, metadata !1, metadata !15, metadata !15, metadata !17,  metadata !15} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !3, metadata !8, metadata !12}
!3 = metadata !{metadata !"0x4\00A\001\0032\0032\000\000\000", metadata !4, null, metadata !5, metadata !6, null, null, null} ; [ DW_TAG_enumeration_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from int]
!4 = metadata !{metadata !"0x29", metadata !22} ; [ DW_TAG_file_type ]
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !7}
!7 = metadata !{metadata !"0x28\00A1\001"} ; [ DW_TAG_enumerator ]
!8 = metadata !{metadata !"0x4\00B\002\0064\0064\000\000\000", metadata !4, null, metadata !9, metadata !10, null, null, null} ; [ DW_TAG_enumeration_type ] [B] [line 2, size 64, align 64, offset 0] [def] [from long unsigned int]
!9 = metadata !{metadata !"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"0x28\00B1\001"} ; [ DW_TAG_enumerator ]
!12 = metadata !{metadata !"0x4\00C\003\0032\0032\000\000\000", metadata !4, null, null, metadata !13, null, null, null} ; [ DW_TAG_enumeration_type ] [C] [line 3, size 32, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14}
!14 = metadata !{metadata !"0x28\00C1\001"} ; [ DW_TAG_enumerator ]
!15 = metadata !{}
!17 = metadata !{metadata !19, metadata !20, metadata !21}
!19 = metadata !{metadata !"0x34\00a\00a\00\004\000\001", null, metadata !4, metadata !3, i32* @a, null} ; [ DW_TAG_variable ]
!20 = metadata !{metadata !"0x34\00b\00b\00\005\000\001", null, metadata !4, metadata !8, i64* @b, null} ; [ DW_TAG_variable ]
!21 = metadata !{metadata !"0x34\00c\00c\00\006\000\001", null, metadata !4, metadata !12, i32* @c, null} ; [ DW_TAG_variable ]
!22 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo/tmp"}

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
!23 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
