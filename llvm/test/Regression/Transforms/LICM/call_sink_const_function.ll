; RUN: llvm-as < %s | opt -basicaa -licm | llvm-dis | grep -C1 sin | grep Out: 
declare double %sin(double)
declare void %foo()

double %test(double %X) {
	br label %Loop

Loop:
	call void %foo()    ;; Unknown effects!

	%A = call double %sin(double %X)   ;; Can still hoist/sink call
	br bool true, label %Loop, label %Out

Out:
	ret double %A
}
