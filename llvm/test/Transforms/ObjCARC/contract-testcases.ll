; RUN: opt -objc-arc-contract -S < %s | FileCheck %s
; rdar://9511608

%0 = type opaque
%1 = type opaque
%2 = type { i64, i64 }
%3 = type { i8*, i8* }
%4 = type opaque

declare %0* @"\01-[NSAttributedString(Terminal) pathAtIndex:effectiveRange:]"(%1*, i8* nocapture, i64, %2*) optsize
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_msgSend_fixup(i8*, %3*, ...)
declare void @objc_release(i8*)
declare %2 @NSUnionRange(i64, i64, i64, i64) optsize
declare i8* @objc_autoreleaseReturnValue(i8*)
declare i8* @objc_autorelease(i8*)
declare i8* @objc_msgSend() nonlazybind

; Don't get in trouble on bugpointed code.

; CHECK: define void @test0(
define void @test0() {
bb:
  %tmp = bitcast %4* undef to i8*
  %tmp1 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %tmp) nounwind
  br label %bb3

bb3:                                              ; preds = %bb2
  br i1 undef, label %bb6, label %bb4

bb4:                                              ; preds = %bb3
  switch i64 undef, label %bb5 [
    i64 9223372036854775807, label %bb6
    i64 0, label %bb6
  ]

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4, %bb4, %bb3
  %tmp7 = phi %4* [ undef, %bb5 ], [ undef, %bb4 ], [ undef, %bb3 ], [ undef, %bb4 ]
  unreachable
}

; When rewriting operands for a phi which has multiple operands
; for the same block, use the exactly same value in each block.

; CHECK: define void @test1(
; CHECK: %0 = bitcast i8* %tmp3 to %0* 
; CHECK: br i1 undef, label %bb7, label %bb7
; CHECK: bb7:
; CHECK: %tmp8 = phi %0* [ %0, %bb ], [ %0, %bb ]
define void @test1() {
bb:
  %tmp = tail call %0* bitcast (i8* ()* @objc_msgSend to %0* ()*)()
  %tmp2 = bitcast %0* %tmp to i8*
  %tmp3 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %tmp2) nounwind
  br i1 undef, label %bb7, label %bb7

bb7:                                              ; preds = %bb6, %bb6, %bb5
  %tmp8 = phi %0* [ %tmp, %bb ], [ %tmp, %bb ]
  unreachable
}
