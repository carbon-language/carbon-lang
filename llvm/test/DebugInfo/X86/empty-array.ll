; RUN: llc -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; <rdar://problem/12566646>

%class.A = type { [0 x i32] }

@a = global %class.A zeroinitializer, align 4

; CHECK:      0x0000002d:   DW_TAG_base_type [3]  
; CHECK-NEXT: DW_AT_name
; CHECK-NEXT: DW_AT_byte_size [DW_FORM_data1]  (0x04)
; CHECK-NEXT: DW_AT_encoding [DW_FORM_data1]   (0x05)

; CHECK:      0x00000034:   DW_TAG_array_type [4] *
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]    (cu + 0x0026 => {0x00000026})

; CHECK:      0x00000039:     DW_TAG_subrange_type [5]
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]  (cu + 0x002d => {0x0000002d})
; CHECK-NOT:  DW_AT_upper_bound

; CHECK:      DW_TAG_member [8]
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]  ( .debug_str[0x0000003f] = "x")
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]  (cu + 0x0034 => {0x00000034})

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, metadata !20, i32 4, metadata !"clang version 3.3 (trunk 169136)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [/Volumes/Sandbox/llvm/t.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, %class.A* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!6 = metadata !{i32 786473, metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786434, metadata !20, null, metadata !"A", i32 1, i64 0, i64 32, i32 0, i32 0, null, metadata !8, i32 0, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 0, align 32, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !14}
!9 = metadata !{i32 786445, metadata !20, metadata !7, metadata !"x", i32 1, i64 0, i64 0, i64 0, i32 1, metadata !10} ; [ DW_TAG_member ] [x] [line 1, size 0, align 0, offset 0] [private] [from ]
!10 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 0, i64 32, i32 0, i32 0, metadata !11, metadata !12, i32 0, i32 0} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786465, i64 0, i64 -1} ; [ DW_TAG_subrange_type ] [unbound]
!14 = metadata !{i32 786478, metadata !7, metadata !"A", metadata !"A", metadata !"", metadata !6, i32 1, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !18, i32 1} ; [ DW_TAG_subprogram ] [line 1] [A]
!15 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17}
!17 = metadata !{i32 786447, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!18 = metadata !{metadata !19}
!19 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!20 = metadata !{metadata !"t.cpp", metadata !"/Volumes/Sandbox/llvm"}
