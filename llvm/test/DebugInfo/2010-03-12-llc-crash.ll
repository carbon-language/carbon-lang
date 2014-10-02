; RUN: llc -O0 < %s -o /dev/null
; llc should not crash on this invalid input.
; PR6588
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @foo() {
entry:
  call void @llvm.dbg.declare(metadata !{i32* undef}, metadata !0, metadata !{metadata !"0x102"})
  ret void
}

!0 = metadata !{metadata !"0x101\00sy\00890\000", metadata !1, metadata !2, metadata !7} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00foo\00892\000\001\000\006\000\000\000", metadata !8, metadata !3, metadata !4, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !8} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\004\00clang 1.1\001\00\000\00\000", metadata !9, metadata !10, metadata !10, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !9, metadata !5, null, metadata !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !"0x29", metadata !9} ; [ DW_TAG_file_type ]
!6 = metadata !{null}
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !9, metadata !5} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"qpainter.h", metadata !"QtGui"}
!9 = metadata !{metadata !"splineeditor.cpp", metadata !"src"}
!10 = metadata !{i32 0}
