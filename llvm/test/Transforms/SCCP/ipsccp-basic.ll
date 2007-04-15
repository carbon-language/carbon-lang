; RUN: llvm-upgrade < %s | llvm-as | opt -ipsccp | llvm-dis | \
; RUN:   grep -v {ret i32 17} | grep -v {ret i32 undef} | not grep ret

implementation

internal int %bar(int %A) {
	%X = add int 1, 2
	ret int %A
}

int %foo() {
	%X = call int %bar(int 17)
	ret int %X
}
