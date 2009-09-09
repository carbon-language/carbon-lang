; RUN: llc < %s -march=arm | grep "subs r" | count 2
; RUN: llc < %s -march=arm | grep "adc r"
; RUN: llc < %s -march=arm | grep "sbc r"  | count 2

define i64 @f1(i64 %a, i64 %b) {
entry:
	%tmp = sub i64 %a, %b
	ret i64 %tmp
}

define i64 @f2(i64 %a, i64 %b) {
entry:
        %tmp1 = shl i64 %a, 1
	%tmp2 = sub i64 %tmp1, %b
	ret i64 %tmp2
}
