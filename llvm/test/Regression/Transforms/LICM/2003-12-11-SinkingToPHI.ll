; RUN: llvm-as < %s | opt -licm | lli

implementation   ; Functions:

int %main() {
entry:
	br label %Loop

Loop:
	br bool true, label %LoopCont, label %Out
LoopCont:
	%X = add int 1, 0
	br bool true, label %Out, label %Loop

Out:
	%V = phi int [ 2, %Loop], [ %X, %LoopCont]
	%V2 = sub int %V, 1
	ret int %V2
}

