; RUN: llc -mtriple=x86_64-redhat-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; Check that we only use DW_AT_low_pc for CU which has non text sections

; CHECK: DW_TAG_compile_unit [1]
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK-NOT: DW_AT_high_pc [DW_FORM_addr]
; CHECK: DW_TAG_subprogram [2]

define void @in_data() nounwind section ".data" {
  ret void, !dbg !5
}

!llvm.dbg.sp = !{!0}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"in_data", metadata !"in_data", metadata !"", metadata !1, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 0, i1 false, void ()* @in_data} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"in_data.c", metadata !"/home/i/test", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 12, metadata !"in_data.c", metadata !"/home/i/test", metadata !"clang version 2.9 (tags/RELEASE_29/final)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!5 = metadata !{i32 1, i32 51, metadata !6, null}
!6 = metadata !{i32 589835, metadata !0, i32 1, i32 50, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
