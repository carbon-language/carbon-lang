; RUN: llc -stop-after block-placement -o - %s | FileCheck %s

target triple = "thumbv6m-none-none"

define i32* @foo(i32* readonly %p0) {
entry:
  %add.ptr = getelementptr inbounds i32, i32* %p0, i32 10
  %arrayidx = getelementptr inbounds i32, i32* %p0, i32 13
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %p0, i32 12
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %p0, i32 11
  %2 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %add, %2
  %3 = load i32, i32* %add.ptr, align 4
  %add5 = add nsw i32 %add3, %3
  tail call void @g(i32 %add5)
  ret i32* %p0
}

declare void @g(i32)

; CHECK-LABEL: name: foo
; CHECK: [[BASE:\$r[0-7]]], {{.*}} tADDi8
; CHECK-NOT: [[BASE]] = tLDMIA_UPD {{.*}} [[BASE]]
; CHECK: tLDMIA killed [[BASE]], {{.*}} def [[BASE]]
