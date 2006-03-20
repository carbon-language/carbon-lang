; RUN: llvm-as < %s | llc -march=x86 | grep 'A(' | wc -l | grep 1
;
; Make sure the common loop invariant _A(reg) is hoisted up to preheader.

%A = internal global [16 x [16 x int]] zeroinitializer, align 32

void %test(int %row, int %N) {
entry:
	%N = cast int %N to uint
	%tmp5 = setgt int %N, 0
	br bool %tmp5, label %cond_true, label %return

cond_true:
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %cond_true ]
	%i.0.0 = cast uint %indvar to int
	%tmp2 = add int %i.0.0, 1
	%tmp = getelementptr [16 x [16 x int]]* %A, int 0, int %row, int %tmp2
	store int 4, int* %tmp
	%tmp5 = add int %i.0.0, 2
	%tmp7 = getelementptr [16 x [16 x int]]* %A, int 0, int %row, int %tmp5
	store int 5, int* %tmp7
	%indvar.next = add uint %indvar, 1
	%exitcond = seteq uint %indvar.next, %N
	br bool %exitcond, label %return, label %cond_true

return:
	ret void
}
