; RUN: llvm-as < %s | opt -ipconstprop -instcombine | llvm-dis | grep 'ret bool true'
implementation

internal int %foo(bool %C) {
	br bool %C, label %T, label %F
T:
	ret int 52
F:
	ret int 52
}

bool %caller(bool %C) {
	%X = call int %foo(bool %C)
	%Y = cast int %X to bool
	ret bool %Y
}
