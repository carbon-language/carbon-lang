; If we have an 'and' of the result of an 'or', and one of the 'or' operands
; cannot have contributed any of the resultant bits, delete the or.  This
; occurs for very common C/C++ code like this:
;
; struct foo { int A : 16; int B : 16; };
; void test(struct foo *F, int X, int Y) {
;        F->A = X; F->B = Y;
; }
;
; Which corresponds to test1.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 'or '

int %test1(int %X, int %Y) {
	%A = and int %X, 7
	%B = and int %Y, 8
	%C = or int %A, %B
	%D = and int %C, 7  ;; This cannot include any bits from %Y!
	ret int %D
}

int %test2(int %X, ubyte %Y) {
	%B = cast ubyte %Y to int
	%C = or int %X, %B
	%D = and int %C, 65536  ;; This cannot include any bits from %Y!
	ret int %D
}

int %test3(int %X, int %Y) {
	%B = shl int %Y, ubyte 1
	%C = or int %X, %B
	%D = and int %C, 1  ;; This cannot include any bits from %Y!
	ret int %D
}

uint %test4(uint %X, uint %Y) {
	%B = shr uint %Y, ubyte 31
	%C = or uint %X, %B
	%D = and uint %C, 2  ;; This cannot include any bits from %Y!
	ret uint %D
}

int %or_test1(int %X, int %Y) {
	%A = and int %X, 1
	%B = or int %A, 1     ;; This cannot include any bits from X!
	ret int %B
}

ubyte %or_test2(ubyte %X, ubyte %Y) {
	%A = shl ubyte %X, ubyte 7
	%B = or ubyte %A, 128     ;; This cannot include any bits from X!
	ret ubyte %B
}

