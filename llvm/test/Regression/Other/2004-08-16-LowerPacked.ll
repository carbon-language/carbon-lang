; RUN: llvm-as < %s | opt -lower-packed | llvm-dis

%foo = uninitialized global <2 x int>;
%bar = uninitialized global <2 x int>;

implementation   ; Functions:

void %main()
{
	%t0 = load <2 x int>* %foo
	%t2 = add <2 x int> %t0, %t0
	%t3 = select bool false, <2 x int> %t0, <2 x int> %t2
	store <2 x int> %t3, <2 x int>* %bar
          
        %c0 = add <2 x int> <int 1, int 1>, %t0
        %c1 = add <2 x int> %t0, <int 0, int 0>
        %c2 = select bool true, <2 x int> <int 1, int 1>, <2 x int> %t0
        store <2 x int> <int 4, int 4>, <2 x int>* %foo
	ret void
}