%AConst    = constant int 123

%Domain = type { sbyte*, int, int*, int, int, int*, %Domain* }

implementation

; Test setting values of different constants in registers.
; 
void "testConsts"(int %N, float %X)
begin
; <label>:0
	%a = add int %N, 1		; 1 should be put in immed field
	%i = add int %N, 12345678	; constant has to be loaded
	%b = add short 4, 3		; one of the operands shd be immed
	%c = add float %X, 0.0		; will this be optimzzed?
	%d = add float %X, 3.1415	; constant has to be loaded
	%f = add uint 4294967295, 10    ; result shd be  9   (not in immed fld)
	%g = add ushort 20, 65535	; result shd be 19 (65536 in immed fld)
	%j = add ushort 65535, 30	; result shd be 29   (not in immed fld)
	%h = add ubyte  40, 255		; result shd be 39   (255 in immed fld)
	%k = add ubyte  255, 50		; result shd be 49   (not in immed fld)
	
	ret void
end

; A SetCC whose result is used should produce instructions to
; compute the boolean value in a register.  One whose result
; is unused will only generate the condition code but not
; the boolean result.
; 
void "unusedBool"(int * %x, int * %y)
begin
; <label>:0				;		[#uses=0]
	seteq int * %x, %y		; <bool>:0	[#uses=1]
	xor bool %0, true		; <bool>:1	[#uses=0]
	setne int * %x, %y		; <bool>:2	[#uses=0]
	ret void
end

; A constant argument to a Phi produces a Cast instruction in the
; corresponding predecessor basic block.  This checks a few things:
; -- phi arguments coming from the bottom of the same basic block
;    (they should not be forward substituted in the machine code!)
; -- code generation for casts of various types
; -- use of immediate fields for integral constants of different sizes
; -- branch on a constant condition
; 
void "mergeConstants"(int * %x, int * %y)
begin
; <label>:0
	br label %Top
Top:
	phi int    [ 0,    %0 ], [ 1,    %Top ], [ 524288, %Next ]
	phi float  [ 0.0,  %0 ], [ 1.0,  %Top ], [ 2.0,    %Next ]
	phi double [ 0.5,  %0 ], [ 1.5,  %Top ], [ 2.5,    %Next ]
	phi bool   [ true, %0 ], [ false,%Top ], [ true,   %Next ]
	br bool true, label %Top, label %Next
Next:
	br label %Top
end



; A constant argument to a cast used only once should be forward substituted
; and loaded where needed, which happens is:
; -- User of cast has no immediate field
; -- User of cast has immediate field but constant is too large to fit
;    or constant is not resolved until later (e.g., global address)
; -- User of cast uses it as a call arg. or return value so it is an implicit
;    use but has to be loaded into a virtual register so that the reg.
;    allocator can allocate the appropriate phys. reg. for it
;  
int* "castconst"(float)
begin
; <label>:0
	%castbig   = cast ulong 99999999 to int
	%castsmall = cast ulong 1        to int
	%usebig    = add int %castbig, %castsmall
		
	%castglob = cast int* %AConst to long*
	%dummyl   = load long* %castglob
	
	%castnull = cast ulong 0 to int*
	ret int* %castnull
end



; Test branch-on-comparison-with-zero, in two ways:
; 1. can be folded
; 2. cannot be folded because result of comparison is used twice
;
void "testbool"(int %A, int %B) {
	br label %Top
Top:
	%D = add int %A, %B
	%E = sub int %D, -4
	%C = setle int %E, 0
	br bool %C, label %retlbl, label %loop

loop:
	%F = add int %A, %B
	%G = sub int %D, -4
	%D = setle int %G, 0
	%E = xor bool %D, true
	br bool %E, label %loop, label %Top

retlbl:
	ret void
end


;; Test use of a boolean result in cast operations.
;; Requires converting a condition code result into a 0/1 value in a reg.
;; 
implementation

int %castbool(int %A, int %B) {
bb0:						; [#uses=0]
    %cond213 = setlt int %A, %B			; <bool> [#uses=1]
    %cast110 = cast bool %cond213 to ubyte      ; <ubyte> [#uses=1]
    %cast109 = cast ubyte %cast110 to int       ; <int> [#uses=1]
    ret int %cast109
}


;; Test use of a boolean result in arithmetic and logical operations.
;; Requires converting a condition code result into a 0/1 value in a reg.
;; 
bool %boolexpr(bool %b, int %N) {
    %b2 = setge int %N, 0
    %b3 = and bool %b, %b2
    ret bool %b3
}


; Test branch on floating point comparison
;
void "testfloatbool"(float %x, float %y)   ; Def %0, %1 - float
begin
; <label>:0
	br label %Top
Top:
	%p = add float %x, %y    ; Def 2 - float
	%z = sub float %x, %y    ; Def 3 - float
	%b = setle float %p, %z	 ; Def 0 - bool
	%c = xor bool %b, true	 ; Def 1 - bool
	br bool %b, label %Top, label %goon
goon:
	ret void
end


; Test cases where an LLVM instruction requires no machine
; instructions (e.g., cast int* to long).  But there are 2 cases:
; 1. If the result register has only a single use and the use is in the
;    same basic block, the operand will be copy-propagated during
;    instruction selection.
; 2. If the result register has multiple uses or is in a different
;    basic block, it cannot (or will not) be copy propagated during
;    instruction selection.  It will generate a
;    copy instruction (add-with-0), but this copy should get coalesced
;    away by the register allocator.
;
int "checkForward"(int %N, int* %A)
begin

bb2:		;;<label>
	%reg114 = shl int %N, ubyte 2		;;
	%cast115 = cast int %reg114 to long	;; reg114 will be propagated
	%cast116 = cast int* %A to long		;; %A will be propagated 
	%reg116  = add long %cast116, %cast115	;;
	%castPtr = cast long %reg116 to int*    ;; %A will be propagated 
	%reg118 = load int* %castPtr		;;
	%cast117 = cast int %reg118 to long	;; reg118 will be copied 'cos
	%reg159 = add long 1234567, %cast117	;;  cast117 has 2 uses, here
	%reg160 = add long 7654321, %cast117	;;  and here.
	ret int 0
end


; Test case for unary NOT operation constructed from XOR.
; 
void "checkNot"(bool %b, int %i)
begin
	%notB = xor bool %b, true
	%notI = xor int %i, -1
	%F    = setge int %notI, 100
	%J    = add int %i, %i
	%andNotB = and bool %F, %notB		;; should get folded with notB
	%andNotI = and int %J, %notI		;; should get folded with notI

	%notB2 = xor bool true, %b		;; should become XNOR
	%notI2 = xor int -1, %i			;; should become XNOR

	ret void
end


; Test case for folding getelementptr into a load/store
;
int "checkFoldGEP"(%Domain* %D, long %idx)
begin
        %reg841 = getelementptr %Domain* %D, long 0, ubyte 1
        %reg820 = load int* %reg841
        ret int %reg820
end
