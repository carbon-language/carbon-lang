; RUN: llvm-as < %s | llvm-dis

%bar = external global <2 x int>		; <<2 x int>*> [#uses=1]

implementation   ; Functions:

void %main() 
{
	store <2 x int> < int 0, int 1 >, <2 x int>* %bar
	ret void
}
