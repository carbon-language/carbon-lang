; this testcase is distilled from this C source:
; int *foo(unsigned N, unsigned M) {
;   unsigned i = (N+1)*sizeof(int);
;   unsigned j = (M+1)*sizeof(int);
;   return (int*)malloc(i+j);
; }

; RUN: as < %s | opt -raise | dis | grep ' cast ' | not grep '*'

implementation

int* %test(uint %N, uint %M) {
	%reg111 = shl uint %N, ubyte 2		; <uint> [#uses=1]
	%reg109 = add uint %reg111, 4		; <uint> [#uses=1]
	%reg114 = shl uint %M, ubyte 2		; <uint> [#uses=1]
	%reg112 = add uint %reg114, 4		; <uint> [#uses=1]
	%reg116 = add uint %reg109, %reg112		; <uint> [#uses=1]
	%reg117 = malloc sbyte, uint %reg116		; <sbyte*> [#uses=1]
	%cast221 = cast sbyte* %reg117 to int*		; <int*> [#uses=1]
	ret int* %cast221
}
