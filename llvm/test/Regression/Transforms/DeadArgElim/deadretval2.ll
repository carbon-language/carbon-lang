; RUN: llvm-as < %s | opt -deadargelim -die | llvm-dis | not grep DEAD

%P = external global int 

implementation


internal int %test(int %DEADARG) {  ; Dead arg only used by dead retval
        ret int %DEADARG
}

internal int %test2(int %DEADARG) {
	%DEADRETVAL = call int %test(int %DEADARG)
	ret int %DEADRETVAL
}

void %test3(int %X) {
	%DEADRETVAL = call int %test2(int %X)
	ret void
}

internal int %foo() {
	%DEAD = load int* %P
	ret int %DEAD
}

internal int %id(int %X) {
	ret int %X
}

void %test4() {
	%DEAD = call int %foo()
	%DEAD2 = call int %id(int %DEAD)
	ret void
}
	
