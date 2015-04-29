; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: @test
; CHECK: ret void, !bar !1, !foo !0
define void @test() {
  add i32 2, 1, !bar !0
  add i32 1, 2, !foo !1
  call void @llvm.dbg.func.start(metadata !"foo")
  extractvalue {{i32, i32}, i32} undef, 0, 1, !foo !0
  ret void, !foo !0, !bar !1
}

; CHECK: define void @test2() !foo !2 !baz !3
define void @test2() !foo !2 !baz !3 {
  unreachable
}

; CHECK: define void @test3() !bar !3
; CHECK: unreachable, !bar !4
define void @test3() !bar !3 {
  unreachable, !bar !4
}

!0 = !DILocation(line: 662302, column: 26, scope: !1)
!1 = !DISubprogram(name: "foo")
!2 = distinct !{}
!3 = distinct !{}
!4 = distinct !{}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

!foo = !{ !0 }
!bar = !{ !1 }
