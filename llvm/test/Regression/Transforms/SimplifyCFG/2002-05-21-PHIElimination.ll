; CFG Simplification is making a loop dead, then changing the add into:
;
;   %V1 = add int %V1, 1
;
; Which is not valid SSA
;
; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis

void "test"() {
	br bool true, label %end, label %Loop

Loop:
	%V = phi int [0, %0], [%V1, %Loop]
	%V1 = add int %V, 1

	br label %Loop
end:
	ret void
}
