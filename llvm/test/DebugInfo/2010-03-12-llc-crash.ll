; RUN: llc -O0 < %s -o /dev/null
; llc should not crash on this invalid input.
; PR6588
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @foo() {
entry:
  call void @llvm.dbg.declare(metadata i32* undef, metadata !0, metadata !{!"0x102"})
  ret void
}

!0 = !{!"0x101\00sy\00890\000", !1, !2, !7} ; [ DW_TAG_arg_variable ]
!1 = !{!"0x2e\00foo\00foo\00foo\00892\000\001\000\006\000\000\000", !8, !3, !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !8} ; [ DW_TAG_file_type ]
!3 = !{!"0x11\004\00clang 1.1\001\00\000\00\000", !9, !10, !10, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = !{!"0x15\00\000\000\000\000\000\000", !9, !5, null, !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = !{!"0x29", !9} ; [ DW_TAG_file_type ]
!6 = !{null}
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", !9, !5} ; [ DW_TAG_base_type ]
!8 = !{!"qpainter.h", !"QtGui"}
!9 = !{!"splineeditor.cpp", !"src"}
!10 = !{i32 0}
