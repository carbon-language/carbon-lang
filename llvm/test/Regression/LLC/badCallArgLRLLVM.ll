; This caused a problem because the argument of a call was defined by
; the return value of another call that appears later in the code.
; When processing the first call, the second call has not yet been processed
; so no LiveRange has been created for its return value.
; 
; llc dies in UltraSparcRegInfo::suggestRegs4CallArgs() with:
;     ERROR: In call instr, no LR for arg: 0x1009e0740 
;
implementation   ; Functions:

declare int %getInt(int);

int %main(int %argc, sbyte** %argv) {
bb0:					;[#uses=0]
        br label %bb2

bb1:
	%reg222 = call int (int)* %getInt(int %reg218) ;; ARG #1 HAS NO LR
	%reg110 = add int %reg222, 1
	%b = setle int %reg110, 0
	br bool %b, label %bb2, label %bb3

bb2:
	%reg218 = call int (int)* %getInt(int %argc)   ;; THIS CALL NOT YET SEEN
	br label %bb1

bb3:
	ret int %reg110
}

