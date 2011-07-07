; RUN: llc < %s -march=arm -mattr=+v4t
; RUN: llc < %s -march=arm -mattr=+v4t | grep cmpne | count 1
; RUN: llc < %s -march=arm -mattr=+v4t | grep bx | count 2

define i32 @t1(i32 %a, i32 %b, i32 %c, i32 %d) {
	switch i32 %c, label %cond_next [
		 i32 1, label %cond_true
		 i32 7, label %cond_true
	]

cond_true:
	%tmp12 = add i32 %a, 1
	%tmp1518 = add i32 %tmp12, %b
	ret i32 %tmp1518

cond_next:
	%tmp15 = add i32 %b, %a
	ret i32 %tmp15
}
