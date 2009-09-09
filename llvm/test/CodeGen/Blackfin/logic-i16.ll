; RUN: llc < %s -march=bfin

define i16 @and(i16 %A, i16 %B) {
	%R = and i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @or(i16 %A, i16 %B) {
	%R = or i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @xor(i16 %A, i16 %B) {
	%R = xor i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}
