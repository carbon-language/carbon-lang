; The "bug" is in the level raising code, not correctly
; raising an array reference.  As generated, the code will work, but does
; not correctly match the array type.  In short, the code generated
; corresponds to this:
;
; int Array[100][200];
; ...
;   Sum += Array[0][i*200+j];
;
; which is out of range, because, although it is correctly accessing the
; array, it does not match types correctly.  LLI would pass it through fine,
; if only the code looked like this:
;  
;   Sum += Array[i][j];
;    
; which is functionally identical, but matches the array bound correctly.
; The fix is to have the -raise pass correctly convert it to the second
; equivelent form.
;
; RUN: as < %s | opt -q -raise > Output/%s.raised.bc
; RUN: lli -force-interpreter -array-checks -abort-on-exception < Output/%s.raised.bc
;

implementation

int "main"()
begin
bb0:					;[#uses=0]
	%Array = alloca [100 x [200 x int]]		; <[100 x [200 x int]] *> [#uses=1]
	%cast1032 = cast [100 x [200 x int]] * %Array to [200 x int] *		; <[200 x int] *> [#uses=1]
	br label %bb1

bb1:					;[#uses=4]
	%cond1033 = setgt long 0, 99		; <bool> [#uses=1]
	br bool %cond1033, label %bb5, label %bb2

bb2:					;[#uses=5]
	%reg124 = phi double [ %reg130, %bb4 ], [ 0.000000e+00, %bb1 ]		; <double> [#uses=2]
	%reg125 = phi int [ %reg131, %bb4 ], [ 0, %bb1 ]		; <int> [#uses=2]
	%cast1043 = cast int %reg125 to int		; <int> [#uses=1]
	%cast1038 = cast int %reg125 to uint		; <uint> [#uses=1]
	%cond1034 = setgt long 0, 199		; <bool> [#uses=1]
	br bool %cond1034, label %bb4, label %bb3

bb3:					;[#uses=5]
	%reg126 = phi double [ %reg128, %bb3 ], [ %reg124, %bb2 ]		; <double> [#uses=1]
	%reg127 = phi int [ %reg129, %bb3 ], [ 0, %bb2 ]		; <int> [#uses=2]
	%cast1042 = cast int %reg127 to int		; <int> [#uses=1]
	%cast1039 = cast int %reg127 to uint		; <uint> [#uses=1]
	%reg110 = mul uint %cast1038, 200		; <uint> [#uses=1]
	%reg111 = add uint %reg110, %cast1039		; <uint> [#uses=1]
	%reg113 = shl uint %reg111, ubyte 2		; <uint> [#uses=1]
	%cast115 = cast uint %reg113 to ulong		; <ulong> [#uses=1]
	%cast1040 = cast [200 x int] * %cast1032 to ulong		; <sbyte *> [#uses=1]
	%reg118 = add ulong %cast1040, %cast115	; <sbyte *> [#uses=1]
	%cast1041 = cast ulong %reg118 to int *		; <int *> [#uses=1]
	%reg120 = load int * %cast1041		; <int> [#uses=1]
	%cast119 = cast int %reg120 to double		; <double> [#uses=1]
	%reg128 = add double %reg126, %cast119		; <double> [#uses=2]
	%reg129 = add int %cast1042, 1		; <int> [#uses=2]
	%cond1035 = setle int %reg129, 199		; <bool> [#uses=1]
	br bool %cond1035, label %bb3, label %bb4

bb4:					;[#uses=5]
	%reg130 = phi double [ %reg128, %bb3 ], [ %reg124, %bb2 ]		; <double> [#uses=2]
	%reg131 = add int %cast1043, 1		; <int> [#uses=2]
	%cond1036 = setle int %reg131, 99		; <bool> [#uses=1]
	br bool %cond1036, label %bb2, label %bb5

bb5:					;[#uses=2]
	%reg132 = phi double [ %reg130, %bb4 ], [ 0.000000e+00, %bb1 ]		; <double> [#uses=1]
	%RET = cast double %reg132 to int
	ret int %RET
end

