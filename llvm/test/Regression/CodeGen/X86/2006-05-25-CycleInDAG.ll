; RUN: llvm-as < %s | llc -march=x86

int %test() {
	br bool false, label %cond_next33, label %cond_true12

cond_true12:
	ret int 0

cond_next33:
	%tmp44.i = call double %foo( double 0.000000e+00, int 32 )
	%tmp61.i = load ubyte* null
	%tmp61.i = cast ubyte %tmp61.i to int
	%tmp58.i = or int 0, %tmp61.i
	%tmp62.i = or int %tmp58.i, 0
	%tmp62.i = cast int %tmp62.i to double
	%tmp64.i = add double %tmp62.i, %tmp44.i
	%tmp68.i = call double %foo( double %tmp64.i, int 0 )
	ret int 0
}

declare double %foo(double, int)
