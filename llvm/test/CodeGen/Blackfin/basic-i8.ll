; RUN: llc < %s -march=bfin

define i8 @add(i8 %A, i8 %B) {
	%R = add i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @sub(i8 %A, i8 %B) {
	%R = sub i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @mul(i8 %A, i8 %B) {
	%R = mul i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @sdiv(i8 %A, i8 %B) {
	%R = sdiv i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @udiv(i8 %A, i8 %B) {
	%R = udiv i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @srem(i8 %A, i8 %B) {
	%R = srem i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @urem(i8 %A, i8 %B) {
	%R = urem i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @and(i8 %A, i8 %B) {
	%R = and i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @or(i8 %A, i8 %B) {
	%R = or i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}

define i8 @xor(i8 %A, i8 %B) {
	%R = xor i8 %A, %B		; <i8> [#uses=1]
	ret i8 %R
}
