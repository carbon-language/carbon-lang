; RUN: llvm-as < %s | llvm-dis

%foo = uninitialized global <4 x float>;
%bar = uninitialized global <4 x float>;

implementation   ; Functions:

void %main()
{
	%t0 = load <4 x float>* %foo
	%t2 = add <4 x float> %t0, %t0
	%t3 = select bool false, <4 x float> %t0, <4 x float> %t2
	store <4 x float> %t3, <4 x float>* %bar  
	ret void
}