; RUN: llvm-as < %s | lli -force-interpreter=false

int %main() {
	br label %Loop
Loop:
	%X = phi int [0, %0], [1, %Loop]
	br bool true, label %Out, label %Loop
Out:
	ret int %X
}
