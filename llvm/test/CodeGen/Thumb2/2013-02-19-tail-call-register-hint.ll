; RUN: llc < %s -mtriple=thumbv7s-apple-ios6.0.0 -verify-machineinstrs

; Check to make sure the tail-call return at the end doesn't use a
; callee-saved register. Register hinting from t2LDRDri was getting this
; wrong. The intervening call will force allocation to try a high register
; first, so the hint will attempt to fire, but must be rejected due to
; not being in the allocation order for the tcGPR register class.
; The machine instruction verifier will make sure that all actually worked
; out the way it's supposed to.

%"myclass" = type { %struct.foo }
%struct.foo = type { i32, [40 x i8] }

define hidden void @func(i8* %Data) nounwind ssp {
  %1 = getelementptr inbounds i8, i8* %Data, i32 12
  %2 = bitcast i8* %1 to %"myclass"*
  tail call void @abc(%"myclass"* %2) nounwind
  tail call void @def(%"myclass"* %2) nounwind
  %3 = getelementptr inbounds i8, i8* %Data, i32 8
  %4 = bitcast i8* %3 to i8**
  %5 = load i8** %4, align 4
  tail call void @ghi(i8* %5) nounwind
  %6 = bitcast i8* %Data to void (i8*)**
  %7 = load void (i8*)** %6, align 4
  %8 = getelementptr inbounds i8, i8* %Data, i32 4
  %9 = bitcast i8* %8 to i8**
  %10 = load i8** %9, align 4
  %11 = icmp eq i8* %Data, null
  br i1 %11, label %14, label %12

; <label>:12                                      ; preds = %0
  %13 = tail call %"myclass"* @jkl(%"myclass"* %2) nounwind
  tail call void @mno(i8* %Data) nounwind
  br label %14

; <label>:14                                      ; preds = %12, %0
  tail call void %7(i8* %10) nounwind
  ret void
}

declare void @mno(i8*)

declare void @def(%"myclass"*)

declare void @abc(%"myclass"*)

declare void @ghi(i8*)

declare %"myclass"* @jkl(%"myclass"*) nounwind
