; RUN: llvm-as < %s | llc -march=ppc32 -enable-ppc-pattern-isel | grep 'fn\?madd\|fn\?msub' | wc -l | grep 5

double %test_FMADD(double %A, double %B, double %C) {
	%D = mul double %A, %B
	%E = add double %D, %C
	ret double %E
}
double %test_FMSUB(double %A, double %B, double %C) {
	%D = mul double %A, %B
	%E = sub double %D, %C
	ret double %E
}
double %test_FNMADD1(double %A, double %B, double %C) {
	%D = mul double %A, %B
	%E = sub double %D, %C
	%F = sub double -0.0, %E
	ret double %F
}
double %test_FNMADD2(double %A, double %B, double %C) {
	%D = mul double %A, %B
	%E = add double %D, %C
	%F = sub double -0.0, %E
	ret double %F
}
double %test_FNMADD3(double %A, double %B, double %C) {
	%D = mul double %A, %B
	%E = add double %C, %D
	%F = sub double -0.0, %E
	ret double %F
}
