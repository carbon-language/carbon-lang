; RUN: llc < %s -march=bfin -verify-machineinstrs

define i64 @add(i64 %A, i64 %B) {
	%R = add i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @sub(i64 %A, i64 %B) {
	%R = sub i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @mul(i64 %A, i64 %B) {
	%R = mul i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @sdiv(i64 %A, i64 %B) {
	%R = sdiv i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @udiv(i64 %A, i64 %B) {
	%R = udiv i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @srem(i64 %A, i64 %B) {
	%R = srem i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @urem(i64 %A, i64 %B) {
	%R = urem i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @and(i64 %A, i64 %B) {
	%R = and i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @or(i64 %A, i64 %B) {
	%R = or i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}

define i64 @xor(i64 %A, i64 %B) {
	%R = xor i64 %A, %B		; <i64> [#uses=1]
	ret i64 %R
}
