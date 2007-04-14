; RUN: llvm-upgrade < %s | llvm-as | opt -constprop | llvm-dis | not grep call

declare double %cos.f64(double)
declare double %sin.f64(double)
declare double %tan.f64(double)
declare double %sqrt.f64(double)
declare bool %llvm.isunordered.f64(double, double)

double %T() {
	%A = call double %cos.f64(double 0.0)
	%B = call double %sin.f64(double 0.0)
	%a = add double %A, %B
	%C = call double %tan.f64(double 0.0)
	%b = add double %a, %C
	%D = call double %sqrt.f64(double 4.0)
	%c = add double %b, %D
	ret double %c
}

bool %TNAN() {
	%A = call bool %llvm.isunordered.f64(double 0x7FF8000000000000, double 1.0)  ;; it's a nan!
	%B = call bool %llvm.isunordered.f64(double 123.0, double 1.0)
	%C = or bool %A, %B
	ret bool %C
}
