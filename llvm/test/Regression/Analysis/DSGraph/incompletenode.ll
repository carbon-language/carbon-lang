; This test is failing because the globals X and Y are marked incomplete
; in the TD graph for %test

; RUN: as < %s | opt -no-aa -ds-aa -load-vn -gcse | dis | not grep seteq

%X = internal global int 20
%Y = internal global int* null

implementation

bool %test(int** %P) { 
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

