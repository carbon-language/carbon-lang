; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis

implementation

int %test(int %X) {
	%Arr = alloca [2 x int]
	%tmp.0 = getelementptr [2 x int]* %Arr, int 0, int 0
	store int 1, int* %tmp.0
	%tmp.1 = getelementptr [2 x int]* %Arr, int 0, int 1
	store int 2, int* %tmp.1

	;; This should turn into a select instruction.
	%tmp.3 = getelementptr [2 x int]* %Arr, int 0, int %X
	%tmp.4 = load int* %tmp.3
	ret int %tmp.4
}
