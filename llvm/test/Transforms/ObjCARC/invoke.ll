; RUN: opt -S -objc-arc < %s | FileCheck %s

declare i8* @objc_retain(i8*)
declare void @objc_release(i8*)
declare i8* @objc_msgSend(i8*, i8*, ...)
declare void @use_pointer(i8*)
declare void @callee()

; ARCOpt shouldn't try to move the releases to the block containing the invoke.

; CHECK: define void @test0(
; CHECK: invoke.cont:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   ret void
; CHECK: lpad:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   ret void
define void @test0(i8* %zipFile) {
entry:
  call i8* @objc_retain(i8* %zipFile) nounwind
  call void @use_pointer(i8* %zipFile)
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*)*)(i8* %zipFile) 
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
           cleanup
  call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
  ret void
}

; ARCOpt should move the release before the callee calls.

; CHECK: define void @test1(
; CHECK: invoke.cont:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   call void @callee()
; CHECK:   br label %done
; CHECK: lpad:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   call void @callee()
; CHECK:   br label %done
; CHECK: done:
; CHECK-NEXT: ret void
define void @test1(i8* %zipFile) {
entry:
  call i8* @objc_retain(i8* %zipFile) nounwind
  call void @use_pointer(i8* %zipFile)
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*)*)(i8* %zipFile)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @callee()
  br label %done

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
           cleanup
  call void @callee()
  br label %done

done:
  call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
  ret void
}

declare i32 @__gxx_personality_v0(...)

!0 = metadata !{}
