; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output

declare int* %bar()

float* %foo() {
	%tmp.11 = invoke float* cast (int* ()* %bar to float* ()*)()
			to label %invoke_cont except label %X

invoke_cont:
	ret float *%tmp.11
X:
	ret float *null
}
