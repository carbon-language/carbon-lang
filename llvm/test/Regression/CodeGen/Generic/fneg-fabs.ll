; RUN: llvm-as < %s | llc

double %fneg(double %X) {
	%Y = sub double -0.0, %X
	ret double %Y
}

float %fnegf(float %X) {
	%Y = sub float -0.0, %X
	ret float %Y
}

declare double %fabs(double)
declare float %fabsf(float)


double %fabstest(double %X) {
	%Y = call double %fabs(double %X)
	ret double %Y
}

float %fabsftest(float %X) {
	%Y = call float %fabsf(float %X)
	ret float %Y
}

