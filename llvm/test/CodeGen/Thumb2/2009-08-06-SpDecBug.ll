; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabi | FileCheck %s
; PR4659
; PR4682

define hidden arm_aapcscc i32 @__gcov_execlp(i8* %path, i8* %arg, ...) nounwind {
entry:
; CHECK: __gcov_execlp:
; CHECK: mov sp, r7
; CHECK: sub sp, #4
	call arm_aapcscc  void @__gcov_flush() nounwind
	br i1 undef, label %bb5, label %bb

bb:		; preds = %bb, %entry
	br i1 undef, label %bb5, label %bb

bb5:		; preds = %bb, %entry
	%0 = alloca i8*, i32 undef, align 4		; <i8**> [#uses=1]
	%1 = call arm_aapcscc  i32 @execvp(i8* %path, i8** %0) nounwind		; <i32> [#uses=1]
	ret i32 %1
}

declare hidden arm_aapcscc void @__gcov_flush()

declare arm_aapcscc i32 @execvp(i8*, i8**) nounwind
