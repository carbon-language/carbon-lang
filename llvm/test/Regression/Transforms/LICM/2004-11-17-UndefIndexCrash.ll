; RUN: llvm-as < %s | opt -licm -disable-output

	%struct.roadlet = type { sbyte*, %struct.vehicle*, [8 x %struct.roadlet*], [8 x %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)*] }
	%struct.vehicle = type { %struct.roadlet*, sbyte*, int, int, %union.._631., int }
	%union.._631. = type { int }

implementation   ; Functions:

declare %struct.roadlet* %_Z11return_nullP7roadletP7vehicle9direction(%struct.roadlet*, %struct.vehicle*, int)

declare %struct.roadlet* %_Z14lane_switch_okP7roadletP7vehicle9direction(%struct.roadlet*, %struct.vehicle*, int)

void %main() {
__main.entry:		; preds = %invoke_cont.1
	br label %invoke_cont.3

invoke_cont.3:		; preds = %__main.entry, %invoke_cont.3
	%tmp.34.i.i502.7 = getelementptr %struct.roadlet* null, int 0, uint 3, int 7		; <%struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)**> [#uses=1]
	store %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)* %_Z11return_nullP7roadletP7vehicle9direction, %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)** %tmp.34.i.i502.7
	store %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)* %_Z14lane_switch_okP7roadletP7vehicle9direction, %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)** null
	%tmp.4.i.i339 = getelementptr %struct.roadlet* null, int 0, uint 3, int undef		; <%struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)**> [#uses=1]
	store %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)* %_Z11return_nullP7roadletP7vehicle9direction, %struct.roadlet* (%struct.roadlet*, %struct.vehicle*, int)** %tmp.4.i.i339
	br label %invoke_cont.3
}
