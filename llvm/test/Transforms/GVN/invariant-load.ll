; Test if the !invariant.load metadata is maintained by GVN.
; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

define i32 @test1(i32* nocapture %p, i8* nocapture %q) {
; CHECK-LABEL: test1
; CHECK: %x = load i32* %p, align 4, !invariant.load !0
; CHECK-NOT: %y = load
entry:
  %x = load i32* %p, align 4, !invariant.load !0
  %conv = trunc i32 %x to i8
  store i8 %conv, i8* %q, align 1
  %y = load i32* %p, align 4, !invariant.load !0
  %add = add i32 %y, 1
  ret i32 %add
}

define i32 @test2(i32* nocapture %p, i8* nocapture %q) {
; CHECK-LABEL: test2
; CHECK-NOT: !invariant.load
; CHECK-NOT: %y = load
entry:
  %x = load i32* %p, align 4
  %conv = trunc i32 %x to i8
  store i8 %conv, i8* %q, align 1
  %y = load i32* %p, align 4, !invariant.load !0
  %add = add i32 %y, 1
  ret i32 %add
}

!0 = metadata !{ }

