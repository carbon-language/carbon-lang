; A store or load cannot alias a global if the accessed amount is larger then
; the global.

; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | llvm-dis | not grep load

%B = global short 8

implementation

short %test(int *%P) {
	%X = load short* %B
	store int 7, int* %P
	%Y = load short* %B
	%Z = sub short %Y, %X
	ret short %Z
}

