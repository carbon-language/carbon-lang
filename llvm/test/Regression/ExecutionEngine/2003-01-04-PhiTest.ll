; RUN: llvm-as -f %s -o %t.bc
; RUN: lli %t.bc > /dev/null

int %main() {
	br label %Loop
Loop:
	%X = phi int [0, %0], [1, %Loop]
	br bool true, label %Out, label %Loop
Out:
	ret int %X
}
