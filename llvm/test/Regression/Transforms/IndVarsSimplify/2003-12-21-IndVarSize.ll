; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep indvar | not grep uint

%G = global long 0

void %test() {
	br label %Loop
Loop:
	%X = phi long [1, %0], [%X.next, %Loop]
	%X.next = add long %X, 1
	store long %X, long* %G
	br label %Loop
}
