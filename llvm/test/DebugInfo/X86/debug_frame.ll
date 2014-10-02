; RUN: llc %s -mtriple=i686-pc-linux-gnu -o - | FileCheck %s

; Test that we produce a .debug_frame, not an .eh_frame

; CHECK: .cfi_sections .debug_frame

define void @f() nounwind {
entry:
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7}
!5 = metadata !{metadata !0}

!0 = metadata !{metadata !"0x2e\00f\00f\00\001\000\001\000\006\00256\001\001", metadata !6, metadata !1, metadata !3, null, void ()* @f, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!1 = metadata !{metadata !"0x29", metadata !6} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 3.0 ()\001\00\000\00\000", metadata !6, metadata !4, metadata !4, metadata !5, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !6, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!6 = metadata !{metadata !"/home/espindola/llvm/test.c", metadata !"/home/espindola/llvm/build"}
!7 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
