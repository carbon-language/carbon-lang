; RUN: llvm-upgrade < %s | llvm-as | opt -loopsimplify -licm -disable-output
void %main() {
entry:
	br bool false, label %Out, label %loop

loop:
	%LI = setgt int 0, 0
	br bool %LI, label %loop, label %Out

Out:
	ret void
}
