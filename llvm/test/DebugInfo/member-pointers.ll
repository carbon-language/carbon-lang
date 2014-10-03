; REQUIRES: object-emission
; XFAIL: hexagon

; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_ptr_to_member_type
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE:0x[0-9a-f]+]]})
; CHECK: [[TYPE]]:   DW_TAG_subroutine_type
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_artificial [DW_FORM_flag
; IR generated from clang -g with the following source:
; struct S {
; };
;
; int S::*x = 0;
; void (S::*y)(int) = 0;

@x = global i64 -1, align 8
@y = global { i64, i64 } zeroinitializer, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16}

!0 = metadata !{metadata !"0x11\004\00clang version 3.3 \000\00\000\00\000", metadata !15, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !1} ; [ DW_TAG_compile_unit ] [/home/blaikie/Development/scratch/simple.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{}
!3 = metadata !{metadata !5, metadata !10}
!5 = metadata !{metadata !"0x34\00x\00x\00\004\000\001", null, metadata !6, metadata !7, i64* @x, null} ; [ DW_TAG_variable ] [x] [line 4] [def]
!6 = metadata !{metadata !"0x29", metadata !15} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x1f\00\000\000\000\000\000", null, null, metadata !8, metadata !9} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from int]
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !"0x13\00S\001\008\008\000\000\000", metadata !15, null, null, metadata !1, null, null, null} ; [ DW_TAG_structure_type ] [S] [line 1, size 8, align 8, offset 0] [def] [from ]
!10 = metadata !{metadata !"0x34\00y\00y\00\005\000\001", null, metadata !6, metadata !11, { i64, i64 }* @y, null} ; [ DW_TAG_variable ] [y] [line 5] [def]
!11 = metadata !{metadata !"0x1f\00\000\000\000\000\000", null, null, metadata !12, metadata !9} ; [ DW_TAG_ptr_to_member_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{null, metadata !14, metadata !8}
!14 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from S]
!15 = metadata !{metadata !"simple.cpp", metadata !"/home/blaikie/Development/scratch"}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
