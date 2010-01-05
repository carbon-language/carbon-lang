; RUN: llvm-as < %s | llvm-dis | grep {ret void, !foo !0, !bar !1}

define void @Foo(i32 %a, i32 %b) {
entry:
  %0 = add i32 %a, 1                         ; <i32> [#uses=1]
  %two = add i32 %b, 2                       ; <i32> [#uses=2]

  call void @llvm.dbg.func.start(metadata !{i32 %0})
  call void @llvm.dbg.func.start(metadata !{i32 %b, i32 %0})
  call void @llvm.dbg.func.start(metadata !{i32 %a, metadata !"foo"})
  call void @llvm.dbg.func.start(metadata !{metadata !0, i32 %two})

  ret void, !foo !0, !bar !1
}

!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4, metadata !"foo"}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

!foo = !{ !0 }
!bar = !{ !1 }
