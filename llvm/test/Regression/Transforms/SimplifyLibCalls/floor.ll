; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*floor(' &&
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep 'call.*floorf('
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*ceil(' &&
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep 'call.*ceilf('
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep 'call.*nearbyint(' &&
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep 'call.*nearbyintf('
; XFAIL: sparc

declare double %floor(double)
declare double %ceil(double)
declare double %nearbyint(double)

float %test_floor(float %C) {
	%D = cast float %C to double
	%E = call double %floor(double %D)  ; --> floorf
	%F = cast double %E to float
	ret float %F
}

float %test_ceil(float %C) {
	%D = cast float %C to double
	%E = call double %ceil(double %D)  ; --> ceilf
	%F = cast double %E to float
	ret float %F
}

float %test_nearbyint(float %C) {
	%D = cast float %C to double
	%E = call double %nearbyint(double %D)  ; --> floorf
	%F = cast double %E to float
	ret float %F
}

