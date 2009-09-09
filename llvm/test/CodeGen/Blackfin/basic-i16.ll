; RUN: llc < %s -march=bfin

define i16 @add(i16 %A, i16 %B) {
	%R = add i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @sub(i16 %A, i16 %B) {
	%R = sub i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @mul(i16 %A, i16 %B) {
	%R = mul i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @sdiv(i16 %A, i16 %B) {
	%R = sdiv i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @udiv(i16 %A, i16 %B) {
	%R = udiv i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @srem(i16 %A, i16 %B) {
	%R = srem i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}

define i16 @urem(i16 %A, i16 %B) {
	%R = urem i16 %A, %B		; <i16> [#uses=1]
	ret i16 %R
}
