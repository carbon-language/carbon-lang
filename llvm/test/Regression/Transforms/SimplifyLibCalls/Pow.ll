; Testcase for calls to the standard C "pow" function
;
; Equivalent to: http://gcc.gnu.org/ml/gcc-patches/2003-02/msg01786.html
; RUN: llvm-as < %s | opt -simplify-libcalls -disable-output &&
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call double .pow'

declare double %pow(double, double)

double %test1(double %X) {
	%Y = call double %pow(double %X, double 0.0)
	ret double %Y    ; x^0.0 always equals 1.0
}

double %test2(double %X) {
	%Y = call double %pow(double %X, double -0.0)
	ret double %Y    ; x^-0.0 always equals 1.0
}

double %test3(double %X) {
	%Y = call double %pow(double 1.0, double %X)
	ret double %Y    ; 1.0^x always equals 1.0
}

