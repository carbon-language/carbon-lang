; RUN: opt < %s -simplify-libcalls -S -mtriple "i386-pc-linux" | FileCheck -check-prefix=DO-SIMPLIFY %s
; RUN: opt < %s -simplify-libcalls -S -mtriple "i386-pc-win32" | FileCheck -check-prefix=DONT-SIMPLIFY %s
; RUN: opt < %s -simplify-libcalls -S -mtriple "x86_64-pc-win32" | FileCheck -check-prefix=C89-SIMPLIFY %s
; RUN: opt < %s -simplify-libcalls -S -mtriple "i386-pc-mingw32" | FileCheck -check-prefix=DO-SIMPLIFY %s
; RUN: opt < %s -simplify-libcalls -S -mtriple "x86_64-pc-mingw32" | FileCheck -check-prefix=DO-SIMPLIFY %s
; RUN: opt < %s -simplify-libcalls -S -mtriple "sparc-sun-solaris" | FileCheck -check-prefix=DO-SIMPLIFY %s

; DO-SIMPLIFY: call float @floorf(
; DO-SIMPLIFY: call float @ceilf(
; DO-SIMPLIFY: call float @roundf(
; DO-SIMPLIFY: call float @nearbyintf(

; C89-SIMPLIFY: call float @floorf(
; C89-SIMPLIFY: call float @ceilf(
; C89-SIMPLIFY: call double @round(
; C89-SIMPLIFY: call double @nearbyint(

; DONT-SIMPLIFY: call double @floor(
; DONT-SIMPLIFY: call double @ceil(
; DONT-SIMPLIFY: call double @round(
; DONT-SIMPLIFY: call double @nearbyint(

declare double @floor(double)

declare double @ceil(double)

declare double @round(double)

declare double @nearbyint(double)

define float @test_floor(float %C) {
	%D = fpext float %C to double		; <double> [#uses=1]
        ; --> floorf
	%E = call double @floor( double %D )		; <double> [#uses=1]
	%F = fptrunc double %E to float		; <float> [#uses=1]
	ret float %F
}

define float @test_ceil(float %C) {
	%D = fpext float %C to double		; <double> [#uses=1]
	; --> ceilf
        %E = call double @ceil( double %D )		; <double> [#uses=1]
	%F = fptrunc double %E to float		; <float> [#uses=1]
	ret float %F
}

define float @test_round(float %C) {
	%D = fpext float %C to double		; <double> [#uses=1]
	; --> roundf
        %E = call double @round( double %D )		; <double> [#uses=1]
	%F = fptrunc double %E to float		; <float> [#uses=1]
	ret float %F
}

define float @test_nearbyint(float %C) {
	%D = fpext float %C to double		; <double> [#uses=1]
	; --> nearbyintf
        %E = call double @nearbyint( double %D )		; <double> [#uses=1]
	%F = fptrunc double %E to float		; <float> [#uses=1]
	ret float %F
}

