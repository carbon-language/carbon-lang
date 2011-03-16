; RUN: llc < %s -march=x86-64 | FileCheck %s

; CHECK: @bar1
; CHECK: movzbl
; CHECK: callq
define void @bar1(i1 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i1 %v1 to i32
  %call = tail call i32 (...)* @foo(i32 %conv) nounwind
  ret void
}

; CHECK: @bar2
; CHECK-NOT: movzbl
; CHECK: callq
define void @bar2(i8 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i8 %v1 to i32
  %call = tail call i32 (...)* @foo(i32 %conv) nounwind
  ret void
}

declare i32 @foo(...)
