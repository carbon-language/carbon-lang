; RUN: llc %s -mtriple=i686-pc-linux-gnu -o - | FileCheck %s

; Test that we produce a .debug_frame, not an .eh_frame

; CHECK: .cfi_sections .debug_frame

define void @f() nounwind {
entry:
  ret void
}

!llvm.dbg.cu = !{!2}
!5 = metadata !{metadata !0}

!0 = metadata !{i32 786478, metadata !6, metadata !1, metadata !"f", metadata !"f", metadata !"", i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, void ()* @f, null, null, null, i32 1} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !6} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !6, i32 12, metadata !"clang version 3.0 ()", i1 true, metadata !"", i32 0, metadata !4, metadata !4, metadata !5, null, null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !6, metadata !1, metadata !"", i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null}
!6 = metadata !{metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/llvm/build"}
