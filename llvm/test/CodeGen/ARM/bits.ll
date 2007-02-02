; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep and      | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=arm | grep orr      | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=arm | grep eor      | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=arm | grep mov.*lsl | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=arm | grep mov.*asr | wc -l | grep 1

define i32 @f1(i32 %a, i32 %b) {
entry:
	%tmp2 = and i32 %b, %a		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @f2(i32 %a, i32 %b) {
entry:
	%tmp2 = or i32 %b, %a		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @f3(i32 %a, i32 %b) {
entry:
	%tmp2 = xor i32 %b, %a		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @f4(i32 %a, i32 %b) {
entry:
	%tmp3 = shl i32 %a, %b		; <i32> [#uses=1]
	ret i32 %tmp3
}

define i32 @f5(i32 %a, i32 %b) {
entry:
	%tmp3 = ashr i32 %a, %b		; <i32> [#uses=1]
	ret i32 %tmp3
}
