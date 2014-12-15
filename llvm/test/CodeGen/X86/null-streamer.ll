; Check the MCNullStreamer operates correctly, at least on a minimal test case.
;
; RUN: llc -filetype=null -o %t -march=x86 %s
; RUN: llc -filetype=null -o %t -mtriple=i686-cygwin %s

define void @f0()  {
  ret void
}

define void @f1() {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !13}

!0 = !{!"0x11\004\00 \001\00\000\00\000", !1, !2, !2, !3, !9, !2} ; [ DW_TAG_compile_unit ]
!1 = !{!"", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00\00\00\002\000\001\000\006\00256\001\002", !1, !5, !6, null, i32 ()* null, null, null, !2} ; [ DW_TAG_subprogram ]
!5 = !{!"0x29", !1} ; [ DW_TAG_file_type ]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ]
!7 = !{!8}
!8 = !{!"0x24\00\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!9 = !{!10}
!10 = !{!"0x34\00i\00i\00_ZL1i\001\001\001", null, !5, !8, null, null} ; [ DW_TAG_variable ]
!11 = !{i32 2, !"Dwarf Version", i32 3}
!13 = !{i32 1, !"Debug Info Version", i32 2}
