; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*floor('

declare double %floor(double)

float %test(float %C) {
	%D = cast float %C to double
	%E = call double %floor(double %D)  ; --> floorf
	%F = cast double %E to float
	ret float %F
}

