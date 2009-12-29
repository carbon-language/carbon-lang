; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | grep {ret void, !bar !1, !foo !0}
define void @test() {
  add i32 2, 1, !bar !0
  add i32 1, 2, !foo !1
  ret void, !foo !0, !bar !1
}

!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4}
