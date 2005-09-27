; RUN: llvm-as < %s | opt -lowerinvoke -disable-output &&
; RUN: llvm-as < %s | opt -lowerinvoke -disable-output -enable-correct-eh-support

implementation

int %foo() {
	invoke int %foo() to label %Ok unwind label %Crap
Ok:
	invoke int %foo() to label %Ok2 unwind label %Crap
Ok2:
	ret int 2
Crap:
	ret int 1
}

int %bar(int %blah) {
	br label %doit
doit:
	;; Value live across an unwind edge.
	%B2 = add int %blah, 1
	invoke int %foo() to label %Ok unwind label %Crap
Ok:
	invoke int %foo() to label %Ok2 unwind label %Crap
Ok2:
	ret int 2
Crap:
	ret int %B2
}
