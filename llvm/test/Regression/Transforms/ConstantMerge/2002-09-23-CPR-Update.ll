; RUN: llvm-as < %s | opt -constmerge > /dev/null

%foo = internal constant {int} {int 7} 
%bar = internal constant {int} {int 7} 

implementation

declare int %test(int*)

void %foo() {
	call int %test(int* getelementptr ( {int} * %foo, long 0, ubyte 0))
	call int %test(int* getelementptr ( {int} * %bar, long 0, ubyte 0))
	ret void
}
