; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -globalsmodref-aa -load-vn -gcse -instcombine | llvm-dis | \
; RUN:   grep {ret i32 0}
; END.
%G = internal global int* null

implementation

void %test() {
	%A = malloc int
	store int* %A, int** %G
	ret void
}

int %test1(int *%P) {
	%g1 = load int** %G
	%h1 = load int* %g1

	; This store cannot alias either G or g1.
	store int 123, int* %P

	%g2 = load int** %G
	%h2 = load int* %g1
	%X = sub int %h1, %h2   ;; -> 0
	ret int %X
}
