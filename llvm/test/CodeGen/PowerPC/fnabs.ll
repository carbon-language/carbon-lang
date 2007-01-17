; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep fnabs

declare double %fabs(double)

implementation

double %test(double %X) {
	%Y = call double %fabs(double %X)
	%Z = sub double -0.0, %Y
	ret double %Z
}
