; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
; PR7105

define void @foo(i32 %x) {
  call void @llvm.zonk(metadata i32 %x, i64 0, metadata !1)
  store i32 0, i32* null, !whatever !0, !whatever_else !{}, !more !{!"hello"}
  store i32 0, i32* null, !whatever !{!"hello", !1, !{}, !2}
  ret void, !_1 !0
}

declare void @llvm.zonk(metadata, i64, metadata) nounwind readnone

!named = !{!0}
!another_named = !{}
!0 = !{i8** null}
!1 = !{i8* null, !2}
!2 = !{}
