; RUN: llc < %s -march=bfin

; Assertion failed: (width < BitWidth && "Invalid APInt Truncate request"),
; function trunc, file APInt.cpp, line 956.

@str2 = external global [29 x i8]

define void @printArgsNoRet(i32 %a1, float %a2, i8 %a3, double %a4, i8* %a5, i32 %a6, float %a7, i8 %a8, double %a9, i8* %a10, i32 %a11, float %a12, i8 %a13, double %a14, i8* %a15) {
entry:
	%tmp17 = sext i8 %a13 to i32
	%tmp23 = call i32 (i8*, ...)* @printf(i8* getelementptr ([29 x i8]* @str2, i32 0, i64 0), i32 %a11, double 0.000000e+00, i32 %tmp17, double %a14, i32 0)
	ret void
}

declare i32 @printf(i8*, ...)
