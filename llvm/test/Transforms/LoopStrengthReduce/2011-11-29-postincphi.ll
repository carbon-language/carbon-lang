; RUN: llc < %s | FileCheck %s
;
; PR11431: handle a phi operand that is replaced by a postinc user.
; LSR first expands %mul to %iv1 in %phi32
; LSR then expands %iv1 in %phi32 into two decrements, one on each loop exit.

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare i1 @check() nounwind

; Check that LSR did something close to the behavior at the time of the bug.
; CHECK: @sqlite3DropTriggerPtr
; CHECK: addq $48, %rax
; CHECK: je
; CHECK: addq $-48, %rax
; CHECK: ret
define i64 @sqlite3DropTriggerPtr() {
entry:
  %cmp0 = call zeroext i1 @check()
  br i1 %cmp0, label %"8", label %"3"

"3":                                              ; preds = %entry
  br i1 %cmp0, label %"4", label %"8"

"4":                                              ; preds = %"3"
  br i1 %cmp0, label %"8", label %"5.preheader"

"5.preheader":                                    ; preds = %"4"
  br label %"5"

"5":
  %iv0 = phi i32 [ %iv0.inc, %"6" ], [ 0, %"5.preheader" ]
  %iv1 = phi i64 [ %iv1.inc, %"6" ], [ 48, %"5.preheader" ]
  %iv0.inc = add nsw i32 %iv0, 1
  %cmp = icmp eq i32 %iv0.inc, 0
  br i1 %cmp, label %"7", label %"6"

"6":
  %iv1.inc = add i64 %iv1, 48
  %iv1.ofs = add i64 %iv1, 40
  %ptr8 = getelementptr inbounds i8* undef, i64 %iv1.ofs
  %ptr32 = bitcast i8* %ptr8 to i32**
  %v = load i32** %ptr32, align 8
  br i1 %cmp0, label %"8", label %"5"

"7":
  %iv.inc64 = sext i32 %iv0.inc to i64
  %mul = mul i64 %iv.inc64, 48
  br label %"8"

"8":                                              ; preds = %"7", %"5", %"4", %"3", %entry
  %phi32 = phi i32 [ %iv0.inc, %"7" ], [ 0, %"4" ], [ 0, %"3" ], [ -1000000, %entry ], [ %iv0.inc, %"6" ]
  %phi64 = phi i64 [ %mul, %"7" ], [ 0, %"4" ], [ 0, %"3" ], [ -48000000, %entry ], [ %iv1, %"6" ]
  ret i64 %phi64
}
