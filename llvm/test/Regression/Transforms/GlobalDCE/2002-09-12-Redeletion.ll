; RUN: llvm-as < %s | opt -globaldce

%foo = internal global int 7         ;; Should die when function %foo is killed

%bar = internal global [2x { int *, int }] [  { int *, int } { int* %foo, int 7}, {int*, int} { int* %foo, int 1 }]

implementation

internal int %foo() {               ;; dies when %b dies.
	%ret = load int* %foo
	ret int %ret
}

