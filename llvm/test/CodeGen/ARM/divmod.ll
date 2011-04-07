; RUN: llc < %s -mtriple=arm-apple-darwin -use-divmod-libcall | FileCheck %s

define void @foo(i32 %x, i32 %y, i32* nocapture %P) nounwind ssp {
entry:
; CHECK: foo:
; CHECK: bl ___divmodsi4
; CHECK-NOT: bl ___divmodsi4
  %div = sdiv i32 %x, %y
  store i32 %div, i32* %P, align 4
  %rem = srem i32 %x, %y
  %arrayidx6 = getelementptr inbounds i32* %P, i32 1
  store i32 %rem, i32* %arrayidx6, align 4
  ret void
}

define void @bar(i32 %x, i32 %y, i32* nocapture %P) nounwind ssp {
entry:
; CHECK: bar:
; CHECK: bl ___udivmodsi4
; CHECK-NOT: bl ___udivmodsi4
  %div = udiv i32 %x, %y
  store i32 %div, i32* %P, align 4
  %rem = urem i32 %x, %y
  %arrayidx6 = getelementptr inbounds i32* %P, i32 1
  store i32 %rem, i32* %arrayidx6, align 4
  ret void
}
