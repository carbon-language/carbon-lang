; RUN: llvm-as < %s | opt -adce -disable-output

int %main() {
	br label %loop

loop:
	br label %loop
}
