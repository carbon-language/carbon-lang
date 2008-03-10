; Testcase for calls to the standard C "pow" function
;
; Equivalent to: http://gcc.gnu.org/ml/gcc-patches/2003-02/msg01786.html
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call double .pow}
; END.

declare double @pow(double, double)

define double @test1(double %X) {
	%Y = call double @pow( double %X, double 0.000000e+00 )		; <double> [#uses=1]
	ret double %Y
}

define double @test2(double %X) {
	%Y = call double @pow( double %X, double -0.000000e+00 )		; <double> [#uses=1]
	ret double %Y
}

define double @test3(double %X) {
	%Y = call double @pow( double 1.000000e+00, double %X )		; <double> [#uses=1]
	ret double %Y
}

