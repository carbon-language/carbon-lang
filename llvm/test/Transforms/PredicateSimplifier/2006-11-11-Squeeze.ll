; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

	%struct.cube_struct = type { int, int, int, int*, int*, int*, int*, int*, uint*, uint*, uint**, uint**, uint*, uint*, uint, int, int*, int, int }
%cube = external global %struct.cube_struct		; <%struct.cube_struct*> [#uses=2]

implementation   ; Functions:

fastcc void %cube_setup() {
entry:
	%tmp = load int* getelementptr (%struct.cube_struct* %cube, int 0, uint 2)		; <int> [#uses=2]
	%tmp = setlt int %tmp, 0		; <bool> [#uses=1]
	br bool %tmp, label %bb, label %cond_next

cond_next:		; preds = %entry
	%tmp2 = load int* getelementptr (%struct.cube_struct* %cube, int 0, uint 1)		; <int> [#uses=2]
	%tmp5 = setlt int %tmp2, %tmp		; <bool> [#uses=1]
	br bool %tmp5, label %bb, label %bb6

bb:		; preds = %cond_next, %entry
	unreachable

bb6:		; preds = %cond_next
	%tmp98124 = setgt int %tmp2, 0		; <bool> [#uses=1]
	br bool %tmp98124, label %bb42, label %bb99

bb42:		; preds = %bb6
	ret void

bb99:		; preds = %bb6
	ret void
}
