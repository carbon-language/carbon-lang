; The loop canonicalization pass should guarantee that there is one backedge 
; for all loops.  This allows the -indvars pass to recognize the %IV 
; induction variable in this testcase.

; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep indvar

int %test(bool %C) {
	br label %Loop
Loop:
	%IV = phi uint [1, %0], [%IV2, %BE1], [%IV2, %BE2]
	%IV2 = add uint %IV, 2
	br bool %C, label %BE1, label %BE2
BE1:
	br label %Loop
BE2:
	br label %Loop
}
