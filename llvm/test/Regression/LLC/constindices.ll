; Test that a sequence of constant indices are folded correctly
; into the equivalent offset at compile-time.

%MixedA = type { float, [15 x int], sbyte, float }

%MixedB = type { float, %MixedA, float }

%fmtArg = internal global [44 x sbyte] c"sqrt(2) = %g\0Aexp(1) = %g\0Api = %g\0Afive = %g\0A\00"; <[44 x sbyte]*> [#uses=1]

implementation

declare int "printf"(sbyte*, ...)

int "main"()
begin
	%ScalarA = alloca %MixedA
	%ScalarB = alloca %MixedB
	%ArrayA  = alloca %MixedA, uint 4
	%ArrayB  = alloca %MixedB, uint 3

	%I1 = getelementptr %MixedA* %ScalarA, long 0, ubyte 0	
	store float 1.4142, float *%I1
	%I2 = getelementptr %MixedB* %ScalarB, long 0, ubyte 1, ubyte 0 
	store float 2.7183, float *%I2
	
  	%fptrA = getelementptr %MixedA* %ArrayA, long 1, ubyte 0 
	%fptrB = getelementptr %MixedB* %ArrayB, long 2, ubyte 1, ubyte 0 
	
	store float 3.1415, float* %fptrA
	store float 5.0,    float* %fptrB
	
	;; Test that a sequence of GEPs with constant indices are folded right
	%fptrA1 = getelementptr %MixedA* %ArrayA, long 3	  ; &ArrayA[3]
	%fptrA2 = getelementptr %MixedA* %fptrA1, long 0, ubyte 1 ; &(*fptrA1).1
	%fptrA3 = getelementptr [15 x int]* %fptrA2, long 0, long 8 ; &(*fptrA2)[8]
	store int 5, int* %fptrA3	; ArrayA[3].1[8] = 5

	%sqrtTwo = load float *%I1
	%exp     = load float *%I2
	%I3 = getelementptr %MixedA* %ArrayA, long 1, ubyte 0 
	%pi      = load float* %I3
	%I4 = getelementptr %MixedB* %ArrayB, long 2, ubyte 1, ubyte 0  
	%five    = load float* %I4
		 
	%dsqrtTwo = cast float %sqrtTwo to double
	%dexp     = cast float %exp to double
	%dpi      = cast float %pi to double
	%dfive    = cast float %five to double
		  
	%castFmt = getelementptr [44 x sbyte]* %fmtArg, long 0, long 0
	call int (sbyte*, ...)* %printf(sbyte* %castFmt, double %dsqrtTwo, double %dexp, double %dpi, double %dfive)
	
	ret int 0
end
