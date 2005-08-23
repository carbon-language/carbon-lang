; RUN: llvm-as < %s | llc -march=c | grep fmod

double %test(double %A, double %B) {
	%C = rem double %A, %B
	ret double %C
}
