; July 6, 2002 -- LLC Regression test
; This test case checks if the integer CC register %xcc (or %ccr)
; is correctly spilled.  The code fragment came from function
; MakeGraph in Olden-mst.
; The original code made all comparisons with 0, so that the %xcc
; register is not needed for the branch in the first basic block.
; Replace 0 with 1 in the first comparson so that the
; branch-on-register instruction cannot be used directly, i.e.,
; the %xcc register is needed for the first branch.
;
	%Graph = type %struct.graph_st*
	%Hash = type %struct.hash*
	%HashEntry = type %struct.hash_entry*
	%Vertex = type %struct.vert_st*
	%struct.graph_st = type { [1 x %Vertex] }
	%struct.hash = type { %HashEntry*, int (uint)*, int }
	%struct.hash_entry = type { uint, sbyte*, %HashEntry }
	%struct.vert_st = type { int, %Vertex, %Hash }
%HashRange = uninitialized global int		; <int*> [#uses=1]
%.LC0 = internal global [13 x sbyte] c"Make phase 2\00"		; <[13 x sbyte]*> [#uses=1]
%.LC1 = internal global [13 x sbyte] c"Make phase 3\00"		; <[13 x sbyte]*> [#uses=1]
%.LC2 = internal global [13 x sbyte] c"Make phase 4\00"		; <[13 x sbyte]*> [#uses=1]
%.LC3 = internal global [15 x sbyte] c"Make returning\00"		; <[15 x sbyte]*> [#uses=1]

implementation   ; Functions:

%Graph %MakeGraph(int %numvert, int %numproc) {
bb1:					;[#uses=1]
	%reg111 = add int %numproc, -1		; <int> [#uses=3]
	%cond275 = setlt int %reg111, 1		; <bool> [#uses=2]
	%cond276 = setle int %reg111, 0		; <bool> [#uses=1]
	%cond277 = setge int %numvert, 0		; <bool> [#uses=2]
	%reg162 = add int %numvert, 3		; <int> [#uses=2]
	br bool %cond275, label %bb7, label %bb4

bb4:
	br bool %cond276, label %bb7, label %bb5

bb5:
	br bool %cond277, label %bb7, label %bb6

bb6:					;[#uses=2]
	ret %Graph null

bb7:					;[#uses=2]
	ret %Graph null
}

