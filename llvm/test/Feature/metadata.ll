; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
; PR7105

define void @foo() {
  call void @llvm.zonk(metadata !1, i64 0, metadata !1)
  ret void
}

declare void @llvm.zonk(metadata, i64, metadata) nounwind readnone

!named = !{!0}
!another_named = !{}
!0 = metadata !{i8** null}
!1 = metadata !{i8* null, metadata !2}
!2 = metadata !{}
