%.LC0 = internal global [10 x sbyte] c"argc: %d\0A\00"

implementation   ; Functions:

declare int %puts(sbyte*)

void %getoptions(int* %argc) {
bb0:		; No predecessors!
	ret void
}

declare int %printf(sbyte*, ...)

int %main(int %argc, sbyte** %argv) {
bb0:		; No predecessors!
	call int (sbyte*, ...)* %printf( sbyte* getelementptr ([10 x sbyte]* %.LC0, long 0, long 0), int %argc)
	%cast224 = cast sbyte** %argv to sbyte*		; <sbyte*> [#uses=1]
	%local = alloca sbyte*		; <sbyte**> [#uses=3]
	store sbyte* %cast224, sbyte** %local
	%cond226 = setle int %argc, 0		; <bool> [#uses=1]
	br bool %cond226, label %bb3, label %bb2

bb2:		; preds = %bb2, %bb0
	%cann-indvar = phi int [ 0, %bb0 ], [ %add1-indvar, %bb2 ]		; <int> [#uses=2]
	%add1-indvar = add int %cann-indvar, 1		; <int> [#uses=2]
	%cann-indvar-idxcast = cast int %cann-indvar to long		; <long> [#uses=1]
	;%reg115 = load sbyte** %local		; <sbyte*> [#uses=1]
	;%cann-indvar-idxcast-scale = mul long %cann-indvar-idxcast, 8		; <long> [#uses=1]
	;%reg232 = getelementptr sbyte* %reg115, long %cann-indvar-idxcast-scale		; <sbyte*> [#uses=1]
	;%cast235 = cast sbyte* %reg232 to sbyte**		; <sbyte**> [#uses=1]
	%CT = cast sbyte**  %local to sbyte***
	%reg115 = load sbyte*** %CT
	%cast235 = getelementptr sbyte** %reg115, long %cann-indvar-idxcast

	%reg117 = load sbyte** %cast235		; <sbyte*> [#uses=1]
	%reg236 = call int %puts( sbyte* %reg117 )		; <int> [#uses=0]
	%cond239 = setlt int %add1-indvar, %argc		; <bool> [#uses=1]
	br bool %cond239, label %bb2, label %bb3

bb3:		; preds = %bb2, %bb0
	%cast243 = cast sbyte** %local to int*		; <int*> [#uses=1]
	call void %getoptions( int* %cast243 )
	ret int 0
}
