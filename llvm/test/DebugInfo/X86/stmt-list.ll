; RUN: llc -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

; CHECK:      .section        .debug_line,"",@progbits
; CHECK-NEXT: .Lsection_line:

; CHECK:      .long   .Lline_table_start0          # DW_AT_stmt_list

define void @f() {
entry:
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7}
!5 = !{!0}

!0 = !{!"0x2e\00f\00f\00\001\000\001\000\006\00256\001\001", !6, !1, !3, null, void ()* @f, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!1 = !{!"0x29", !6} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 3.0 ()\001\00\000\00\000", !6, !4, !4, !5, null, null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !6, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!6 = !{!"test2.c", !"/home/espindola/llvm"}
!7 = !{i32 1, !"Debug Info Version", i32 2}
