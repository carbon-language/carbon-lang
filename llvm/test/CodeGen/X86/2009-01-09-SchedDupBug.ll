; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2
; rdar://6481994

	%Value = type { i32 (...)** }

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind

define %Value* @bar(%Value** %exception) nounwind {
prologue:
	br i1 true, label %NextVerify41, label %FailedVerify

NextVerify41:		; preds = %prologue
	br i1 true, label %NextVerify, label %FailedVerify

NextVerify:		; preds = %NextVerify41
	br i1 false, label %label12, label %label

label:		; preds = %NextVerify
	br i1 true, label %xxNumberLiteral.exit, label %handle_exception

xxNumberLiteral.exit:		; preds = %label
	%0 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 0, i32 0)		; <{ i32, i1 }> [#uses=2]
	%intAdd = extractvalue { i32, i1 } %0, 0		; <i32> [#uses=1]
	%intAddOverflow = extractvalue { i32, i1 } %0, 1		; <i1> [#uses=1]
	%toint55 = ashr i32 %intAdd, 1		; <i32> [#uses=1]
	%toFP56 = sitofp i32 %toint55 to double		; <double> [#uses=1]
	br i1 %intAddOverflow, label %exit, label %label12

label12:		; preds = %xxNumberLiteral.exit, %NextVerify
	%var_lr1.0 = phi double [ %toFP56, %xxNumberLiteral.exit ], [ 0.000000e+00, %NextVerify ]		; <double> [#uses=0]
	unreachable

exit:		; preds = %xxNumberLiteral.exit
	ret %Value* null

FailedVerify:		; preds = %NextVerify41, %prologue
	ret %Value* null

handle_exception:		; preds = %label
	ret %Value* undef
}
