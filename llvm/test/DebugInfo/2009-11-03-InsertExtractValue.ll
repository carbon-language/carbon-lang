; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!dbg = !{!0}
!0 = metadata !{i32 786478, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", metadata !1, i32 3, metadata !2, i1 false, i1 false, i32 0, i32 0, null, i32 258, i1 false, null, null, i32 0, metadata !1, i32 3} 
!1 = metadata !{i32 41, metadata !"/foo", metadata !"bar.cpp"} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 21, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !3, i32 0, null} ; [ DW_TAG_subroutine_type ]
!3 = metadata !{null}

define <{i32, i32}> @f1() {
; CHECK: !dbgx !1
  %r = insertvalue <{ i32, i32 }> zeroinitializer, i32 4, 1, !dbgx !1
; CHECK: !dbgx !1
  %e = extractvalue <{ i32, i32 }> %r, 0, !dbgx !1
  ret <{ i32, i32 }> %r
}

; CHECK: [protected]
