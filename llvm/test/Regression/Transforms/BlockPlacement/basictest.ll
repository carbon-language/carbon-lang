; RUN: llvm-as < %s | opt -block-placement -disable-output -print

int %test() {

	br bool true, label %X, label %Y
A:
	ret int 0
X:
	br label %A
Y:
	br label %A
}
