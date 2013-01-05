; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!dbg = !{!0}
!0 = metadata !{i32 786478, i32 0, metadata !1, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", metadata !1, i32 3, metadata !"nard", i1 false, i1 false, i32 0, i32 0, null, i32 258, i1 false, null, null, i32 0, metadata !1, i32 3} 
!1 = metadata !{i32 42}

define <{i32, i32}> @f1() {
; CHECK: !dbgx !1
  %r = insertvalue <{ i32, i32 }> zeroinitializer, i32 4, 1, !dbgx !1
; CHECK: !dbgx !1
  %e = extractvalue <{ i32, i32 }> %r, 0, !dbgx !1
  ret <{ i32, i32 }> %r
}

; CHECK: [protected]
