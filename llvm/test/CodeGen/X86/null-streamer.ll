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

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !" ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !9, metadata !2, metadata !""}
!1 = metadata !{metadata !"", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"", metadata !"", metadata !"", i32 2, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 ()* null, null, null, metadata !2, i32 2}
!5 = metadata !{i32 786473, metadata !1}
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null}
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786484, i32 0, null, metadata !"i", metadata !"i", metadata !"_ZL1i", metadata !5, i32 1, metadata !8, i32 1, i32 1, null, null}
!11 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
