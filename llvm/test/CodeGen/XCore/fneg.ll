; RUN: llc < %s -march=xcore | grep "xor" | count 1
define i1 @test(double %F) nounwind {
entry:
	%0 = fsub double -0.000000e+00, %F
	%1 = fcmp olt double 0.000000e+00, %0
	ret i1 %1
}
