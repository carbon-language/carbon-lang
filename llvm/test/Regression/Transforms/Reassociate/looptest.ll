; This testcase comes from this C fragment:
;
; void test(unsigned Num, int *Array) {
;  unsigned i, j, k;
;
;  for (i = 0; i != Num; ++i)
;    for (j = 0; j != Num; ++j)
;      for (k = 0; k != Num; ++k)
;        printf("%d\n", i+k+j);    /* Reassociate to (i+j)+k */
;}
;
; In this case, we want to reassociate the specified expr so that i+j can be
; hoisted out of the inner most loop.
;
; RUN: as < %s | opt -reassociate | dis | grep 115 | not grep 117

%.LC0 = internal global [4 x sbyte] c"%d\0A\00"		; <[4 x sbyte]*> [#uses=1]

declare int "printf"(sbyte*, ...)

void "test"(uint %Num, int* %Array) {
bb0:					;[#uses=1]
	%cond221 = seteq uint 0, %Num		; <bool> [#uses=3]
	br bool %cond221, label %bb7, label %bb2

bb2:					;[#uses=3]
	%reg115 = phi uint [ %reg120, %bb6 ], [ 0, %bb0 ]		; <uint> [#uses=2]
	br bool %cond221, label %bb6, label %bb3

bb3:					;[#uses=3]
	%reg116 = phi uint [ %reg119, %bb5 ], [ 0, %bb2 ]		; <uint> [#uses=2]
	br bool %cond221, label %bb5, label %bb4

bb4:					;[#uses=3]
	%reg117 = phi uint [ %reg118, %bb4 ], [ 0, %bb3 ]		; <uint> [#uses=2]
	%reg113 = add uint %reg115, %reg117		; <uint> [#uses=1]
	%reg114 = add uint %reg113, %reg116		; <uint> [#uses=1]
	%cast227 = getelementptr [4 x sbyte]* %.LC0, long 0, long 0		; <sbyte*> [#uses=1]
	call int (sbyte*, ...)* %printf( sbyte* %cast227, uint %reg114 )		; <int>:0 [#uses=0]
	%reg118 = add uint %reg117, 1		; <uint> [#uses=2]
	%cond224 = setne uint %reg118, %Num		; <bool> [#uses=1]
	br bool %cond224, label %bb4, label %bb5

bb5:					;[#uses=3]
	%reg119 = add uint %reg116, 1		; <uint> [#uses=2]
	%cond225 = setne uint %reg119, %Num		; <bool> [#uses=1]
	br bool %cond225, label %bb3, label %bb6

bb6:					;[#uses=3]
	%reg120 = add uint %reg115, 1		; <uint> [#uses=2]
	%cond226 = setne uint %reg120, %Num		; <bool> [#uses=1]
	br bool %cond226, label %bb2, label %bb7

bb7:					;[#uses=2]
	ret void
}
