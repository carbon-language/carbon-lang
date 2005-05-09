; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep 'br'
declare void %bar(int)

void %test(bool %P, int* %Q) {
	br bool %P, label %T, label %F
T:
	store int 1, int* %Q
	%A = load int* %Q
	call void %bar(int %A)
	ret void
F:
	store int 1, int* %Q
	%B = load int* %Q
	call void %bar(int %B)
	ret void
}

