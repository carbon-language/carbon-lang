; RUN: llvm-as < %s | llvm-dis

%foo = global <2 x int> <int 0, int 1>;
%bar = uninitialized global <2 x int>;

implementation   ; Functions:

void %main()
{
	%t0 = load <2 x int>* %foo;
	store <2 x int> %t0, <2 x int>* %bar   
	ret void
}