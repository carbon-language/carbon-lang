; Simple sanity check testcase.  Both alloca's should be eliminated.
; RUN: as < %s | opt -mem2reg | dis | not grep 'alloca'

implementation

double "testfunc"(int %i, double %j)
begin
	%I = alloca int
	%J = alloca double

	store int %i, int* %I
	store double %j, double* %J

	%t1 = load int* %I
	%t2 = add int %t1, 1
	store int %t2, int* %I

	%t3 = load int* %I
	%t4 = cast int %t3 to double
	%t5 = load double* %J
	%t6 = mul double %t4, %t5

	ret double %t6
end
