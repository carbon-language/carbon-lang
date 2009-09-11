; RUN: opt < %s -scalarrepl -S | not grep alloca
; RUN: opt < %s -scalarrepl -S | not grep {7 x double}
; RUN: opt < %s -scalarrepl -instcombine -S | grep {ret double %B}

define double @test(double %A, double %B) {
	%ARR = alloca [7 x i64]
	%C = bitcast [7 x i64]* %ARR to double*
	store double %A, double* %C

	%D = getelementptr [7 x i64]* %ARR, i32 0, i32 4
	%E = bitcast i64* %D to double*
	store double %B, double* %E

	%F = getelementptr double* %C, i32 4
	%G = load double* %F
	ret double %G
}


