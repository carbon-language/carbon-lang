; RUN: llvm-as < %s | llvm-dis | llvm-as > /dev/null

%v4f = type <4 x float>

%foo = uninitialized global %v4f
%bar = uninitialized global %v4f

implementation   ; Functions:

void %main() {
	br label %A
C:
	store %v4f %t2, %v4f* %bar  
	ret void

B:
	%t2 = add %v4f %t0, %t0
	br label %C

A:
	%t0 = load %v4f* %foo
	br label %B
}
