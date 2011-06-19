; RUN: llc < %s -march=bfin -verify-machineinstrs > %t

	%0 = type { i24, i1 }		; type %0

define i1 @func2(i24 zeroext %v1, i24 zeroext %v2) nounwind {
entry:
	%t = call %0 @llvm.uadd.with.overflow.i24(i24 %v1, i24 %v2)		; <%0> [#uses=1]
	%obit = extractvalue %0 %t, 1		; <i1> [#uses=1]
	br i1 %obit, label %carry, label %normal

normal:		; preds = %entry
	ret i1 true

carry:		; preds = %entry
	ret i1 false
}

declare %0 @llvm.uadd.with.overflow.i24(i24, i24) nounwind
