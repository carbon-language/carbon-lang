; RUN: llvm-as < %s | opt -instcombine -disable-output

declare int* %bar()

float* %foo() {
	%tmp.11 = invoke float* cast (int* ()* %bar to float* ()*)()
			to label %invoke_cont except label %invoke_cont

invoke_cont:
	ret float *%tmp.11
}
