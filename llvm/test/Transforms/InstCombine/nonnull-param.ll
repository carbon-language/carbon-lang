; RUN: opt -S -instcombine < %s | FileCheck %s

declare void @callee(i8* %arg)

; Positive test - arg is known non null
define void @test(i8* nonnull %arg) {
; CHECK-LABEL: @test
; CHECK: call void @callee(i8* nonnull %arg)
  call void @callee(i8* %arg)
  ret void
}


; Negative test - arg is not known to be non null
define void @test2(i8* %arg) {
; CHECK-LABEL: @test2
; CHECK: call void @callee(i8* %arg)
  call void @callee(i8* %arg)
  ret void
}

declare void @callee2(i8*, i8*, i8*)

; Sanity check arg indexing
define void @test3(i8* %arg1, i8* nonnull %arg2, i8* %arg3) {
; CHECK-LABEL: @test3
; CHECK: call void @callee2(i8* %arg1, i8* nonnull %arg2, i8* %arg3)
  call void @callee2(i8* %arg1, i8* %arg2, i8* %arg3)
  ret void
}

; Because of the way CallSite::paramHasAttribute looks at the callee 
; directly, we will not set the attribute on the CallSite.  That's 
; fine as long as all consumers use the same check. 
define void @test4(i8* nonnull %arg) {
; CHECK-LABEL: @test4
; CHECK: call void @test4(i8* %arg)
  call void @test4(i8* %arg)
  ret void
}


