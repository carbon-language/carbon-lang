implementation

int "main"()
begin
bb0:
	%reg109 = malloc int, uint 100
	br label %bb2

bb2:
	%cann-indvar1 = phi int [ 0, %bb0 ], [ %add1-indvar1, %bb2 ]
	%reg127 = mul int %cann-indvar1, 2
	%add1-indvar1 = add int %cann-indvar1, 1
	store int 999, int * %reg109
	%cond1015 = setle int 1, 99
	%reg128 = add int %reg127, 2
	br bool %cond1015, label %bb2, label %bb4

bb4:					;[#uses=3]
	%cann-indvar = phi uint [ %add1-indvar, %bb4 ], [ 0, %bb2 ]
	%add1-indvar = add uint %cann-indvar, 1		; <uint> [#uses=1]
	store int 333, int * %reg109
	%reg131 = add uint %add1-indvar, 3		; <int> [#uses=1]
	%cond1017 = setle uint %reg131, 99		; <bool> [#uses=1]
	br bool %cond1017, label %bb4, label %bb5

bb5:
	ret int 0
end
