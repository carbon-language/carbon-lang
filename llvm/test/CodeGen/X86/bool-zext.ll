; RUN: llc < %s -march=x86-64 | FileCheck %s

; CHECK: @bar1
; CHECK: movzbl
; CHECK: callq
define void @bar1(i1 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i1 %v1 to i32
  %call = tail call i32 (...)* @foo1(i32 %conv) nounwind
  ret void
}

; CHECK: @bar2
; CHECK-NOT: movzbl
; CHECK: callq
define void @bar2(i8 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i8 %v1 to i32
  %call = tail call i32 (...)* @foo1(i32 %conv) nounwind
  ret void
}

; CHECK: @bar3
; CHECK: callq
; CHECK-NOT: movzbl
; CHECK-NOT: and
; CHECK: ret
define zeroext i1 @bar3() nounwind ssp {
entry:
  %call = call i1 @foo2() nounwind
  ret i1 %call
}

declare i32 @foo1(...)
declare zeroext i1 @foo2()
