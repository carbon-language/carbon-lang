; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

; We can't remove the load because new operators are overideable and can return non-undefined memory.
;CHECK: main
;CHECK: load
;CHECK: ret
define i32 @main(i32 %argc, i8** nocapture %argv) ssp uwtable {
  %1 = tail call noalias i8* @_Znam(i64 800)
  %2 = bitcast i8* %1 to i32**
  %3 = load i32** %2, align 8, !tbaa !0
  %4 = icmp eq i32* %3, null
  %5 = zext i1 %4 to i32
  ret i32 %5
}

declare noalias i8* @_Znam(i64)

!0 = metadata !{metadata !"any pointer", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
