; RUN: llvm-as < %s | opt -constprop | llvm-dis | not grep call

declare double %cos(double)
declare double %sin(double)
declare double %tan(double)
declare double %sqrt(double)
declare bool %llvm.isunordered(double, double)

double %T() {
	%A = call double %cos(double 0.0)
	%B = call double %sin(double 0.0)
	%a = add double %A, %B
	%C = call double %tan(double 0.0)
	%b = add double %a, %C
	%D = call double %sqrt(double 4.0)
	%c = add double %b, %D
	ret double %c
}

bool %TNAN() {
	%A = call bool %llvm.isunordered(double 0x7FF8000000000000, double 1.0)  ;; it's a nan!
	%B = call bool %llvm.isunordered(double 123.0, double 1.0)
	%C = or bool %A, %B
	ret bool %C
}
