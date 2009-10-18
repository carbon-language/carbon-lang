; RUN: llc < %s -mtriple=i386-apple-darwin10 | FileCheck %s

; PR4958

define i32 @main() nounwind ssp {
entry:
; CHECK: main:
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  br label %bb

bb:                                               ; preds = %bb1, %entry
; CHECK:      addl $1
; CHECK-NEXT: adcl $0
  %i.0 = phi i64 [ 0, %entry ], [ %0, %bb1 ]      ; <i64> [#uses=1]
  %0 = add nsw i64 %i.0, 1                        ; <i64> [#uses=2]
  %1 = icmp sgt i32 0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  %2 = icmp sle i64 %0, 1                         ; <i1> [#uses=1]
  br i1 %2, label %bb, label %bb2

bb2:                                              ; preds = %bb1, %bb
  br label %return

return:                                           ; preds = %bb2
  ret i32 0
}
