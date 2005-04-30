; RUN: llvm-as < %s | llc

;; SQRT
declare float %llvm.sqrt(float)
declare double %llvm.sqrt(double)
double %test_sqrt(float %F) {
	%G = call float %llvm.sqrt(float %F)
	%H = cast float %G to double
	%I = call double %llvm.sqrt(double %H)
	ret double %I
}

; SIN
declare float %sinf(float)
declare double %sin(double)
double %test_sin(float %F) {
	%G = call float %sinf(float %F)
	%H = cast float %G to double
	%I = call double %sin(double %H)
	ret double %I
}

; COS
declare float %cosf(float)
declare double %cos(double)
double %test_cos(float %F) {
	%G = call float %cosf(float %F)
	%H = cast float %G to double
	%I = call double %cos(double %H)
	ret double %I
}
