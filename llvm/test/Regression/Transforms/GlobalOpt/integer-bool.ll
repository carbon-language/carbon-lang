; RUN: llvm-as < %s | opt -globalopt -instcombine | llvm-dis | grep 'ret bool true'

;; check that global opt turns integers that only hold 0 or 1 into bools.

%G = internal global int 0    ;; This only holds 0 or 1.

implementation

void %set1() {
	store int 0, int* %G
	ret void
}
void %set2() {
	store int 1, int* %G
	ret void
}

bool %get() {
	%A = load int* %G
	%C = setlt int %A, 2  ;; always true
	ret bool %C
}
