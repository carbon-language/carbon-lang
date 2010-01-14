; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define void @Foo(i32 %a, i32 %b) {
entry:
  %0 = add i32 %a, 1                              ; <i32> [#uses=1]
  %two = add i32 %b, %0                           ; <i32> [#uses=0]
  %1 = alloca i32                                 ; <i32*> [#uses=1]
  %three = bitcast i32* %1 to { }*                ; <{ }*> [#uses=6]

  call void @llvm.dbg.declare({ }* %three, metadata !{i32* %1})
; CHECK: metadata !{i32* %1}
  call void @llvm.dbg.declare({ }* %three, metadata !{{ }* %three})
  call void @llvm.dbg.declare({ }* %three, metadata !{i32 %0})
  call void @llvm.dbg.declare({ }* %three, metadata !{{ }* %three, i32 %0})
  call void @llvm.dbg.declare({ }* %three, metadata !{i32 %b, i32 %0})
  call void @llvm.dbg.declare({ }* %three, metadata !{i32 %a, metadata !"foo"})
; CHECK: metadata !{i32 %a, metadata !"foo"}
  call void @llvm.dbg.declare({ }* %three, metadata !{metadata !0, i32 %two})

  ret void, !foo !0, !bar !1
; CHECK: ret void, !foo !0, !bar !1
}

!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4, metadata !"foo"}

declare void @llvm.dbg.declare({ }*, metadata) nounwind readnone

!foo = !{ !0 }
!bar = !{ !1 }
