; Testcase for calls to the standard C "pow" function
; RUN: if as < %s | opt -instcombine | dis | grep 'call double %pow'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

declare double %pow(double, double)

double %test1(double %X) {
	%Y = call double %pow(double %X, double 0.0)
	ret double %Y    ; x^0.0 always equals 0.0
}

double %test2(double %X) {
	%Y = call double %pow(double %X, double -0.0)
	ret double %Y    ; x^-0.0 always equals 0.0
}

double %test3(double %X) {
	%Y = call double %pow(double 1.0, double %X)
	ret double %Y    ; 1.0^x always equals 1.0
}

