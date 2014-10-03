; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!llvm.dbg.sp = !{!0}
!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!6}

!0 = metadata !{metadata !"0x2e\00bar\00bar\00_ZN3foo3barEv\003\000\000\000\006\00258\000\003", metadata !4, metadata !1, metadata !2, null, null, null, i32 0, metadata !1} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !4} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !4, metadata !1, null, metadata !3, null} ; [ DW_TAG_subroutine_type ]
!3 = metadata !{null}
!4 = metadata !{metadata !"/foo", metadata !"bar.cpp"}
!5 = metadata !{metadata !"0x11\0012\00\001\00\000\00\000", metadata !4, metadata !3, metadata !3, null, null, null}; [DW_TAG_compile_unit ]

define <{i32, i32}> @f1() {
; CHECK: !dbgx ![[NUMBER:[0-9]+]]
  %r = insertvalue <{ i32, i32 }> zeroinitializer, i32 4, 1, !dbgx !1
; CHECK: !dbgx ![[NUMBER]]
  %e = extractvalue <{ i32, i32 }> %r, 0, !dbgx !1
  ret <{ i32, i32 }> %r
}

; CHECK: [protected]
!6 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
