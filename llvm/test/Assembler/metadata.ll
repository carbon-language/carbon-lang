; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | grep {ret void, !foo !0}
define void @test() {
  ret void, !foo !0
;, !bar !1
}

!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4}
