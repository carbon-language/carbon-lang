; RUN: llvm-as < %s | opt -ipconstprop -disable-output
implementation

internal void %foo(int %X) {
	call void %foo(int %X)
	ret void
}
