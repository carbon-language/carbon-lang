; This test makes sure that memmove instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 'call void %llvm.memmove'

%S = internal constant [33 x sbyte] c"panic: restorelist inconsistency\00"

implementation

declare void %llvm.memmove(sbyte*, sbyte*, uint, uint)

void %test1(sbyte* %A, sbyte* %B, uint %N) {
 	;; 0 bytes -> noop.
	call void %llvm.memmove(sbyte* %A, sbyte* %B, uint 0, uint 1)
	ret void
}

void %test2(sbyte *%A, uint %N) {
 	;; dest can't alias source since we can't write to source!
	call void %llvm.memmove(sbyte* %A, sbyte* getelementptr ([33 x sbyte]* %S, int 0, int 0), 
                                uint %N, uint 1)
	ret void
}
