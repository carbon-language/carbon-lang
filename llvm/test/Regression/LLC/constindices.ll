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
	
	store float 1.4142, %MixedA* %ScalarA, uint 0, ubyte 0 
	store float 2.7183, %MixedB* %ScalarB, uint 0, ubyte 1, ubyte 0 
	
  	%fptrA = getelementptr %MixedA* %ArrayA, uint 1, ubyte 0 
	%fptrB = getelementptr %MixedB* %ArrayB, uint 2, ubyte 1, ubyte 0 
	
	store float 3.1415, float* %fptrA
	store float 5.0,    float* %fptrB
	
	%sqrtTwo = load %MixedA* %ScalarA, uint 0, ubyte 0 
	%exp     = load %MixedB* %ScalarB, uint 0, ubyte 1, ubyte 0 
	%pi      = load %MixedA* %ArrayA, uint 1, ubyte 0 
	%five    = load %MixedB* %ArrayB, uint 2, ubyte 1, ubyte 0  
		 
	%dsqrtTwo = cast float %sqrtTwo to double
	%dexp     = cast float %exp to double
	%dpi      = cast float %pi to double
	%dfive    = cast float %five to double
		  
	%castFmt = getelementptr [44 x sbyte]* %fmtArg, uint 0, uint 0
	call int (sbyte*, ...)* %printf(sbyte* %castFmt, double %dsqrtTwo, double %dexp, double %dpi, double %dfive)
	
	ret int 0
end
