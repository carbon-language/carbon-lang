; Check that stack alignment can be forced. Individual targets should test their
; specific implementation details.

; RUN: llc < %s -stackrealign | FileCheck %s
; CHECK-LABEL: @f
; CHECK-LABEL: @g

define i32 @f(i8* %p) nounwind {
entry:
  %0 = load i8, i8* %p
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

define i64 @g(i32 %i) nounwind {
entry:
  br label %if.then

if.then:
  %0 = alloca i8, i32 %i
  call void @llvm.memset.p0i8.i32(i8* %0, i8 0, i32 %i, i1 false)
  %call = call i32 @f(i8* %0)
  %conv = sext i32 %call to i64
  ret i64 %conv
}

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1) nounwind

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"override-stack-alignment", i32 32}
