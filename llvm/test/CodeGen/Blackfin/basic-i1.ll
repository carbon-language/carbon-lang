; RUN: llc < %s -march=bfin > %t

define i1 @add(i1 %A, i1 %B) {
	%R = add i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @sub(i1 %A, i1 %B) {
	%R = sub i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @mul(i1 %A, i1 %B) {
	%R = mul i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @sdiv(i1 %A, i1 %B) {
	%R = sdiv i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @udiv(i1 %A, i1 %B) {
	%R = udiv i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @srem(i1 %A, i1 %B) {
	%R = srem i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @urem(i1 %A, i1 %B) {
	%R = urem i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @and(i1 %A, i1 %B) {
	%R = and i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @or(i1 %A, i1 %B) {
	%R = or i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}

define i1 @xor(i1 %A, i1 %B) {
	%R = xor i1 %A, %B		; <i1> [#uses=1]
	ret i1 %R
}
