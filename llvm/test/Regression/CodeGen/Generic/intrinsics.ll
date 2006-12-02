; RUN: llvm-upgrade < %s | llvm-as | llc

;; SQRT
declare float %llvm.sqrt.f32(float)
declare double %llvm.sqrt.f64(double)
double %test_sqrt(float %F) {
	%G = call float %llvm.sqrt.f32(float %F)
	%H = cast float %G to double
	%I = call double %llvm.sqrt.f64(double %H)
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
