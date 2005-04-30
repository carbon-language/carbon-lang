; RUN: llvm-as < %s | llc


declare float %llvm.sqrt(float)
declare double %llvm.sqrt(double)


double %test_sqrt(float %F) {
	%G = call float %llvm.sqrt(float %F)
	%H = cast float %G to double
	%I = call double %llvm.sqrt(double %H)
	ret double %I
}
