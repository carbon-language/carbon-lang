; Test that a negative constant smaller than 64 bits (e.g., int)
; is correctly implemented with sign-extension.
; In particular, the current code generated is:
;
; main:
; .L_main_LL_0:
;         save    %o6, -224, %o6
;         setx    .G_fmtArg_1, %o1, %o0
;         setuw   1, %o1		! i = 1
;         setuw   4294967295, %o3	! THE BUG: 0x00000000ffffffff
;         setsw   0, %i0
;         add     %i6, 1999, %o2	! fval
;         add     %o1, %g0, %o1
;         add     %o0, 0, %o0
;         mulx    %o1, %o3, %o1		! ERROR: 0xffffffff; should be -1
;         add     %o1, 3, %o1		! ERROR: 0x100000002; should be 0x2
;         mulx    %o1, 12, %o3		! 
;         add     %o2, %o3, %o3		! produces bad address!
;         call    printf
;         nop     
;         jmpl    %i7+8, %g0
;         restore %g0, 0, %g0
; 
;   llc produces:
; ioff = 2        fval = 0xffffffff7fffec90       &fval[2] = 0xb7fffeca8
;   instead of:
; ioff = 2        fval = 0xffffffff7fffec90       &fval[2] = 0xffffffff7fffeca8
; 

%Results = type { float, float, float }

%fmtArg = internal global [39 x sbyte] c"ioff = %u\09fval = 0x%p\09&fval[2] = 0x%p\0A\00"; <[39 x sbyte]*> [#uses=1]

implementation

declare int "printf"(sbyte*, ...)

int "main"()
begin
	%fval   = alloca %Results, uint 4
	%i      = add uint 1, 0					; i = 1
	%iscale = mul uint %i, 4294967295			; i*-1 = -1
	%ioff   = add uint %iscale, 3				; 3+(-i) = 2
	%ioff   = cast uint %ioff to long
	%fptr   = getelementptr %Results* %fval, long %ioff	; &fval[2]
	%castFmt = getelementptr [39 x sbyte]* %fmtArg, long 0, long 0
	call int (sbyte*, ...)* %printf(sbyte* %castFmt, uint %ioff, %Results* %fval, %Results* %fptr)
	ret int 0
end
