; This test was failing because the globals X and Y are marked incomplete
; in the TD graph for %test

; RUN: llvm-as < %s | opt -no-aa -ds-aa -load-vn -gcse -instcombine | llvm-dis | not grep seteq

%X = internal global int 20
%Y = internal global int* null

implementation

internal bool %test(int** %P) { 
  %A = load int** %P              ;; We know P == Y!
  %B = load int** %Y              ;; B = A
  %c = seteq int* %A, %B          ;; Always true
  ret bool %c
}

int %main() {
	store int* %X, int** %Y
	call bool %test(int** %Y)
	ret int 0
}

