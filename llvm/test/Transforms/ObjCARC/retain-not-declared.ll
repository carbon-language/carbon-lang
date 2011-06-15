; RUN: opt -S -objc-arc -objc-arc-contract < %s | FileCheck %s

; Test that the optimizer can create an objc_retainAutoreleaseReturnValue
; declaration even if no objc_retain declaration exists.
; rdar://9401303

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
declare i8* @objc_unretainedObject(i8*)
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_autoreleaseReturnValue(i8*)

; CHECK:      define i8* @foo(i8* %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call i8* @objc_retainAutoreleaseReturnValue(i8* %p) nounwind
; CHECK-NEXT:   ret i8* %0
; CHECK-NEXT: }

define i8* @foo(i8* %p) {
entry:
  %call = tail call i8* @objc_unretainedObject(i8* %p)
  %0 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %1 = tail call i8* @objc_autoreleaseReturnValue(i8* %0) nounwind
  ret i8* %1
}

