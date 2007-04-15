; RUN: llvm-upgrade < %s | llvm-as | opt -ipconstprop -instcombine | \
; RUN:    llvm-dis | grep {ret i1 true}
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
	%Y = setne int %X, 0
	ret bool %Y
}
