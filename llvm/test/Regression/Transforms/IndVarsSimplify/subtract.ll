; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep indvar

%G = global long 0

void %test(long %V) {
	br label %Loop
Loop:
	%X = phi long [1, %0], [%X.next, %Loop]
	%X.next = sub long %X, %V
	store long %X, long* %G
	br label %Loop
}
