; RUN: llvm-as < %s | llc -march=ppc32 -enable-ppc-pattern-isel | not grep fneg

double %test_fneg_sel(double %A, double %B, double %C) {
    %D = sub double -0.0, %A
    %Cond = setgt double %D, -0.0
    %E = select bool %Cond, double %B, double %C
	ret double %E
}
