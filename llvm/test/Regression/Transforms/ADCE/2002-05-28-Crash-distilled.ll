; This testcase is a distilled form of: 2002-05-28-Crash.ll

; RUN: llvm-as < %s | opt -adce 

float "test"(int %i) {
	%F = cast int %i to float    ; This BB is not dead
	%I = cast int %i to uint     ; future dead inst
	br label %Loop

Loop:                                ; This block is dead
	%B = cast uint %I to bool
	br bool %B, label %Out, label %Loop

Out:
	ret float %F
}

