implementation

int %main() {
	call int %mylog(int 4)
	ret int 0
}

internal int %mylog(int %num) {
bb0:            ; No predecessors!
	br label %bb2

bb2:
        %reg112 = phi int [ 10, %bb2 ], [ 1, %bb0 ]
        %cann-indvar = phi int [ %cann-indvar, %bb2 ], [0, %bb0]
        %reg114 = add int %reg112, 1
        %cond222 = setlt int %reg114, %num
        br bool %cond222, label %bb2, label %bb3

bb3:            ; preds = %bb2, %bb0
	ret int %reg114
}

