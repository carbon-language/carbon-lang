; RUN: llvm-as < %s | opt -extract-blocks -disable-output
int %foo() {
	br label %EB
EB:
	%V = invoke int %foo() to label %Cont unwind label %Unw
Cont:
	ret int %V
Unw:
	unwind
}
