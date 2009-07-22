; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_unsigned(i64 %x, i64 %y) {
; CHECK: %z = nuw add i64 %x, %y
	%z = nuw add i64 %x, %y
	ret i64 %z
}

define i64 @sub_unsigned(i64 %x, i64 %y) {
; CHECK: %z = nuw sub i64 %x, %y
	%z = nuw sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_unsigned(i64 %x, i64 %y) {
; CHECK: %z = nuw mul i64 %x, %y
	%z = nuw mul i64 %x, %y
	ret i64 %z
}

define i64 @add_signed(i64 %x, i64 %y) {
; CHECK: %z = nsw add i64 %x, %y
	%z = nsw add i64 %x, %y
	ret i64 %z
}

define i64 @sub_signed(i64 %x, i64 %y) {
; CHECK: %z = nsw sub i64 %x, %y
	%z = nsw sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_signed(i64 %x, i64 %y) {
; CHECK: %z = nsw mul i64 %x, %y
	%z = nsw mul i64 %x, %y
	ret i64 %z
}

define i64 @add_plain(i64 %x, i64 %y) {
; CHECK: %z = add i64 %x, %y
	%z = add i64 %x, %y
	ret i64 %z
}

define i64 @sub_plain(i64 %x, i64 %y) {
; CHECK: %z = sub i64 %x, %y
	%z = sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_plain(i64 %x, i64 %y) {
; CHECK: %z = mul i64 %x, %y
	%z = mul i64 %x, %y
	ret i64 %z
}

define i64 @add_both(i64 %x, i64 %y) {
; CHECK: %z = nuw nsw add i64 %x, %y
	%z = nuw nsw add i64 %x, %y
	ret i64 %z
}

define i64 @sub_both(i64 %x, i64 %y) {
; CHECK: %z = nuw nsw sub i64 %x, %y
	%z = nuw nsw sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_both(i64 %x, i64 %y) {
; CHECK: %z = nuw nsw mul i64 %x, %y
	%z = nuw nsw mul i64 %x, %y
	ret i64 %z
}

define i64 @add_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = nuw nsw add i64 %x, %y
	%z = nsw nuw add i64 %x, %y
	ret i64 %z
}

define i64 @sub_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = nuw nsw sub i64 %x, %y
	%z = nsw nuw sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = nuw nsw mul i64 %x, %y
	%z = nsw nuw mul i64 %x, %y
	ret i64 %z
}

define i64 @sdiv_exact(i64 %x, i64 %y) {
; CHECK: %z = exact sdiv i64 %x, %y
	%z = exact sdiv i64 %x, %y
	ret i64 %z
}

define i64 @sdiv_plain(i64 %x, i64 %y) {
; CHECK: %z = sdiv i64 %x, %y
	%z = sdiv i64 %x, %y
	ret i64 %z
}

define i64 @add_both_ce() {
; CHECK: ret i64 nuw nsw add (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw nuw add (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_both_ce() {
; CHECK: ret i64 nuw nsw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw nuw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_both_ce() {
; CHECK: ret i64 nuw nsw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nuw nsw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sdiv_exact_ce() {
; CHECK: ret i64 exact sdiv (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 exact sdiv (i64 ptrtoint (i64* @addr to i64), i64 91)
}
