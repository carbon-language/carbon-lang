; RUN: llvm-as < %s | opt -basicaa -licm | llvm-dis | %prcontext strlen 1 | grep Out: 
declare int %strlen(sbyte*)
declare void %foo()

int %test(sbyte* %P) {
	br label %Loop

Loop:
	%A = call int %strlen(sbyte* %P)   ;; Can hoist/sink call
	br bool false, label %Loop, label %Out

Out:
	ret int %A
}
