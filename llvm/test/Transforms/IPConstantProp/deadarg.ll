; RUN: llvm-upgrade < %s | llvm-as | opt -ipconstprop -disable-output
implementation

internal void %foo(int %X) {
	call void %foo(int %X)
	ret void
}
