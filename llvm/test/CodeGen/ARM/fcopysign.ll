; RUN: llvm-as < fcopysign.ll | llc -march=arm &&
; RUN: llvm-as < fcopysign.ll | llc -march=arm | grep bic | wc -l | grep 2 &&
; RUN: llvm-as < fcopysign.ll | llc -march=arm -mattr=+v6,+vfp2 &&
; RUN: llvm-as < fcopysign.ll | llc -march=arm -mattr=+v6,+vfp2 | grep fneg | wc -l | grep 2

define float %test1(float %x, double %y) {
	%tmp = fpext float %x to double
	%tmp2 = tail call double %copysign( double %tmp, double %y )
	%tmp2 = fptrunc double %tmp2 to float
	ret float %tmp2
}

define double %test2(double %x, float %y) {
	%tmp = fpext float %y to double
	%tmp2 = tail call double %copysign( double %x, double %tmp )
	ret double %tmp2
}

declare double %copysign(double, double)
