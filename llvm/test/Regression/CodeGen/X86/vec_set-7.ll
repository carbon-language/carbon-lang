; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movq | wc -l | grep 1

<2 x long> %test(<2 x long>* %p) {
	%tmp = cast <2 x long>* %p to double*
	%tmp = load double* %tmp
	%tmp = insertelement <2 x double> undef, double %tmp, uint 0
	%tmp5 = insertelement <2 x double> %tmp, double 0.000000e+00, uint 1
	%tmp = cast <2 x double> %tmp5 to <2 x long>
	ret <2 x long> %tmp
}
