; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep "subs r" | count 2
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep adc
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep sbc

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
