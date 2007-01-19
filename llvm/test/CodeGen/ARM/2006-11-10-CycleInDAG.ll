; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6

	%struct.layer_data = type { int, [2048 x ubyte], ubyte*, [16 x ubyte], uint, ubyte*, int, int, [64 x int], [64 x int], [64 x int], [64 x int], int, int, int, int, int, int, int, int, int, int, int, int, [12 x [64 x short]] }
%ld = external global %struct.layer_data*

void %main() {
entry:
	br bool false, label %bb169.i, label %cond_true11

bb169.i:
        ret void

cond_true11:
	%tmp.i32 = load %struct.layer_data** %ld
	%tmp3.i35 = getelementptr %struct.layer_data* %tmp.i32, int 0, uint 1, int 2048
	%tmp.i36 = getelementptr %struct.layer_data* %tmp.i32, int 0, uint 2
	store ubyte* %tmp3.i35, ubyte** %tmp.i36
	store ubyte* %tmp3.i35, ubyte** null
	ret void
}
