; RUN: llvm-upgrade < %s | llvm-as | opt -adce -disable-output

int %main() {
	br label %loop

loop:
	br label %loop
}
