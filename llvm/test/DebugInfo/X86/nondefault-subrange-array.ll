; RUN: llc -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

%class.A = type { [42 x i32] }

@a = global %class.A zeroinitializer, align 4

; Check that we can handle non-default array bounds. In this case, the array
; goes from [-3, 38].

; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_member
; CHECK-NEXT:                   DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "x")
; CHECK-NEXT:                   DW_AT_type [DW_FORM_ref4]       (cu + 0x{{[0-9a-f]*}} => {[[ARRAY:0x[0-9a-f]*]]})

; CHECK: [[ARRAY]]: DW_TAG_array_type [{{.*}}] *
; CHECK-NEXT:                 DW_AT_type [DW_FORM_ref4]    (cu + 0x{{[0-9a-f]*}} => {[[BASE:0x[0-9a-f]*]]})

; CHECK: DW_TAG_subrange_type
; CHECK-NEXT:                   DW_AT_type [DW_FORM_ref4]  (cu + 0x{{[0-9a-f]*}} => {[[BASE2:0x[0-9a-f]*]]})
; CHECK-NEXT:                   DW_AT_lower_bound [DW_FORM_data8]       (0xfffffffffffffffd)
; CHECK-NEXT:                   DW_AT_count [DW_FORM_data1]       (0x2a)

; CHECK: [[BASE]]: DW_TAG_base_type
; CHECK: [[BASE2]]: DW_TAG_base_type
; CHECK-NEXT:                 DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "sizetype")
; CHECK-NEXT:                 DW_AT_byte_size [DW_FORM_data1] (0x08)
; CHECK-NEXT:                 DW_AT_encoding [DW_FORM_data1]  (DW_ATE_unsigned)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{metadata !"0x11\004\00clang version 3.3 (trunk 169136)\000\00\000\00\000", metadata !20, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !1} ; [ DW_TAG_compile_unit ] [/Volumes/Sandbox/llvm/t.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x34\00a\00a\00\001\000\001", null, metadata !6, metadata !7, %class.A* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!6 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x2\00A\001\000\0032\000\000\000", metadata !20, null, null, metadata !8, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 0, align 32, offset 0] [def] [from ]
!8 = metadata !{metadata !9, metadata !14}
!9 = metadata !{metadata !"0xd\00x\001\000\000\000\001", metadata !20, metadata !7, metadata !10} ; [ DW_TAG_member ] [x] [line 1, size 0, align 0, offset 0] [private] [from ]
!10 = metadata !{metadata !"0x1\00\000\000\0032\000\000", null, null, metadata !11, metadata !12, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!11 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x21\00-3\0042"} ; [ DW_TAG_subrange_type ] [-3, 39]
!14 = metadata !{metadata !"0x2e\00A\00A\00\001\000\000\000\006\00320\000\001", metadata !6, metadata !7, metadata !15, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ] [line 1] [A]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17}
!17 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!18 = metadata !{metadata !19}
!19 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!20 = metadata !{metadata !"t.cpp", metadata !"/Volumes/Sandbox/llvm"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
