; Here we have a case where there are two loops and LICM is hoisting an 
; instruction from one loop into the other loop!  This is obviously bad and 
; happens because preheader insertion doesn't insert a preheader for this
; case... bad.

; RUN: llvm-as < %s | opt -licm -adce -simplifycfg | llvm-dis | not grep 'br '

int %main(int %argc) {
        br label %bb5

bb5:            ; preds = %bb5, %0
	%I = phi int [0, %0], [%I2, %bb5]
	%I2 = add int %I, 1
	%c = seteq int %I2, 10
        br bool %c, label %bb5, label %bb8

bb8:            ; preds = %bb8, %bb5
        %cann-indvar = phi uint [ 0, %bb8 ], [ 0, %bb5 ]
	%X = add int %argc, %argc  ; Loop invariant
        br bool false, label %bb8, label %bb10

bb10:           ; preds = %bb8
        ret int %X
}

