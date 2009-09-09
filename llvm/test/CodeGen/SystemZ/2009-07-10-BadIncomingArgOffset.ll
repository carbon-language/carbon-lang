; RUN: llc < %s | grep 168

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"

declare void @rdft(i32 signext, i32 signext, double*, i32* nocapture, double*) nounwind

declare double @mp_mul_d2i_test(i32 signext, i32 signext, double* nocapture) nounwind

define void @mp_mul_radix_test_bb3(i32 %radix, i32 %nfft, double* %tmpfft, i32* %ip, double* %w, double* %arrayidx44.reload, double* %call.out) nounwind {
newFuncRoot:
	br label %bb3

bb4.exitStub:		; preds = %bb3
	store double %call, double* %call.out
	ret void

bb3:		; preds = %newFuncRoot
	tail call void @rdft(i32 signext %nfft, i32 signext -1, double* %arrayidx44.reload, i32* %ip, double* %w) nounwind
	%call = tail call double @mp_mul_d2i_test(i32 signext %radix, i32 signext %nfft, double* %tmpfft)		; <double> [#uses=1]
	br label %bb4.exitStub
}
