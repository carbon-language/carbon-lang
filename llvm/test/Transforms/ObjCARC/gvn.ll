; RUN: opt -S -basicaa -objc-arc-aa -gvn < %s | FileCheck %s

@x = common global i8* null, align 8

declare i8* @objc_retain(i8*)
declare i32 @objc_sync_start(i8*)
declare i32 @objc_sync_stop(i8*)

; GVN should be able to eliminate this redundant load, with ARC-specific
; alias analysis.

; CHECK: define i8* @test0(i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT: %s = load i8** @x
; CHECK-NOT: load
; CHECK: ret i8* %s
; CHECK-NEXT: }
define i8* @test0(i32 %n) nounwind {
entry:
  %s = load i8** @x
  %0 = tail call i8* @objc_retain(i8* %s) nounwind
  %t = load i8** @x
  ret i8* %t
}

; CHECK: define i8* @test1(i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT: %s = load i8** @x
; CHECK-NEXT: call i32 @objc_sync_start
; CHECK-NEXT: call i32 @objc_sync_stop
; CHECK-NEXT: ret i8* %s
; CHECK-NEXT: }
define i8* @test1(i32 %n) nounwind {
entry:
  %s = load i8** @x
  %0 = call i32 @objc_sync_start(i8* %s)
  %t = load i8** @x
  %1 = call i32 @objc_sync_stop(i8* %s)
  ret i8* %t
}
