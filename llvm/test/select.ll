implementation

; A SetCC whose result is used should produce instructions to
; compute the boolean value in a register.  One whose result
; is unused will only generate the condition code but not
; the boolean result.
; 
void "unusedBool"(int * %x, int * %y)
begin
; <label>:0				;		[#uses=0]
	seteq int * %x, %y		; <bool>:0	[#uses=1]
	not bool %0			; <bool>:1	[#uses=0]
	setne int * %x, %y		; <bool>:2	[#uses=0]
	ret void
end

; A constant argument to a Phi produces a Cast instruction in the
; corresponding predecessor basic block.  This has little to do with
; selection but the code is a bit weird.
; 
void "mergeConstants"(int * %x, int * %y)
begin
; <label>:0						;	  [#uses=1]
	br label %Top
Top:							;	  [#uses=4]
	phi int [ 0, %0 ], [ 1, %Top ], [ 2, %Next ]	; <int>:0 [#uses=0]
	br bool true, label %Top, label %Next
Next:							;	  [#uses=2]
	br label %Top
end


; Test branch-on-comparison-with-zero, in two ways:
; 1. can be folded
; 2. cannot be folded because result of comparison is used twice
;
void "testbool"(int, int)   ; Def %0, %1
	int 0          ; Def 2
	int -4         ; Def 3
begin
	add int %0, %1    ; Def 4
	sub int %4, %3    ; Def 5
	setle int %5, %2  ; Def 0 - bool plane
	br bool %0, label %retlbl, label %loop

loop:
	add int %0, %1    ; Def 6
	sub int %4, %3    ; Def 7
	setle int %7, %2  ; Def 1 - bool
	not bool %1		  ; Def 2 - bool. first use of bool %1
	br bool %1, label %loop, label %0    ;  second use of bool %1

retlbl:
	ret void
end


; Test branch on floating point comparison
;
void "testfloatbool"(float %x, float %y)   ; Def %0, %1 - float
begin
	%p = add float %x, %y    ; Def 2 - float
	%z = sub float %x, %y    ; Def 3 - float
	%b = setle float %p, %z	 ; Def 0 - bool
	%c = not bool %b	 ; Def 1 - bool
	br bool %b, label %0, label %goon
goon:
	ret void
end


; Test cases where an LLVM instruction requires no machine
; instructions (e.g., cast int* to long).  But there are 2 cases:
; 1. If the result register has only a single use, the operand will be
;    copy-propagated during instruction selection.
; 2. If the result register has multiple uses, it cannot be copy 
;    propagated during instruction selection.  It will generate a
;    copy instruction (add-with-0), but this copy should get coalesced
;    away by the register allocator.
;
int "checkForward"(int %N, int* %A)
begin

bb2:		;;<label>
	%reg114 = shl int %N, ubyte 2		;;
	%cast115 = cast int %reg114 to int*	;; reg114 will be propagated
	%reg116 = add int* %A, %cast115		;;
	%reg118 = load int* %reg116		;;
	%cast117 = cast int %reg118 to long	;; reg118 will be copied 'cos
	%reg159 = add long 1234567, %cast117	;;  cast117 has 2 uses, here
	%reg160 = add long 7654321, %cast117	;;  and here.
	ret void
end
