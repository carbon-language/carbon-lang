; RUN: llvm-as < %s | opt -sccp -disable-output

implementation

declare int %foo()

void %caller() {
	br bool true, label %T, label %F
F:
	%X = invoke int %foo() to label %T unwind label %T

T:
	ret void
}
