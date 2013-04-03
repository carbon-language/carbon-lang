; RUN: opt -objc-arc -S < %s | FileCheck %s

declare i8* @objc_release(i8* %x)
declare i8* @objc_retain(i8* %x)
declare i8* @objc_autorelease(i8* %x)
declare i8* @objc_autoreleaseReturnValue(i8* %x)
declare i8* @objc_retainAutoreleasedReturnValue(i8* %x)

; Never tail call objc_autorelease.
define i8* @test0(i8* %x) {
entry:
  ; CHECK: %tmp0 = call i8* @objc_autorelease(i8* %x)
  %tmp0 = call i8* @objc_autorelease(i8* %x)
  ; CHECK: %tmp1 = call i8* @objc_autorelease(i8* %x)
  %tmp1 = tail call i8* @objc_autorelease(i8* %x)

  ret i8* %x
}

; Always tail call autoreleaseReturnValue.
define i8* @test1(i8* %x) {
entry:
  ; CHECK: %tmp0 = tail call i8* @objc_autoreleaseReturnValue(i8* %x)
  %tmp0 = call i8* @objc_autoreleaseReturnValue(i8* %x)
  ; CHECK: %tmp1 = tail call i8* @objc_autoreleaseReturnValue(i8* %x)
  %tmp1 = tail call i8* @objc_autoreleaseReturnValue(i8* %x)
  ret i8* %x
}

; Always tail call objc_retain.
define i8* @test2(i8* %x) {
entry:
  ; CHECK: %tmp0 = tail call i8* @objc_retain(i8* %x)
  %tmp0 = call i8* @objc_retain(i8* %x)
  ; CHECK: %tmp1 = tail call i8* @objc_retain(i8* %x)
  %tmp1 = tail call i8* @objc_retain(i8* %x)
  ret i8* %x
}

define i8* @tmp(i8* %x) {
  ret i8* %x
}

; Always tail call objc_retainAutoreleasedReturnValue.
define i8* @test3(i8* %x) {
entry:
  %y = call i8* @tmp(i8* %x)
  ; CHECK: %tmp0 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %y)
  %tmp0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %y)
  %z = call i8* @tmp(i8* %x)
  ; CHECK: %tmp1 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %z)
  %tmp1 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %z)
  ret i8* %x
}

; By itself, we should never change whether or not objc_release is tail called.
define i8* @test4(i8* %x) {
entry:
  ; CHECK: %tmp0 = call i8* @objc_release(i8* %x)
  %tmp0 = call i8* @objc_release(i8* %x)
  ; CHECK: %tmp1 = tail call i8* @objc_release(i8* %x)
  %tmp1 = tail call i8* @objc_release(i8* %x)
  ret i8* %x
}

; If we convert a tail called @objc_autoreleaseReturnValue to an
; @objc_autorelease, ensure that the tail call is removed.
define i8* @test5(i8* %x) {
entry:
  ; CHECK: %tmp0 = call i8* @objc_autorelease(i8* %x)
  %tmp0 = tail call i8* @objc_autoreleaseReturnValue(i8* %x)
  ret i8* %tmp0
}

