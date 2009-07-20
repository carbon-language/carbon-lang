; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_signed(i64 %x, i64 %y) {
; CHECK: %z = signed add i64 %x, %y
	%z = signed add i64 %x, %y
	ret i64 %z
}

define i64 @sub_signed(i64 %x, i64 %y) {
; CHECK: %z = signed sub i64 %x, %y
	%z = signed sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_signed(i64 %x, i64 %y) {
; CHECK: %z = signed mul i64 %x, %y
	%z = signed mul i64 %x, %y
	ret i64 %z
}

define i64 @add_unsigned(i64 %x, i64 %y) {
; CHECK: %z = unsigned add i64 %x, %y
	%z = unsigned add i64 %x, %y
	ret i64 %z
}

define i64 @sub_unsigned(i64 %x, i64 %y) {
; CHECK: %z = unsigned sub i64 %x, %y
	%z = unsigned sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_unsigned(i64 %x, i64 %y) {
; CHECK: %z = unsigned mul i64 %x, %y
	%z = unsigned mul i64 %x, %y
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
; CHECK: %z = unsigned signed add i64 %x, %y
	%z = unsigned signed add i64 %x, %y
	ret i64 %z
}

define i64 @sub_both(i64 %x, i64 %y) {
; CHECK: %z = unsigned signed sub i64 %x, %y
	%z = unsigned signed sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_both(i64 %x, i64 %y) {
; CHECK: %z = unsigned signed mul i64 %x, %y
	%z = unsigned signed mul i64 %x, %y
	ret i64 %z
}

define i64 @add_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = unsigned signed add i64 %x, %y
	%z = signed unsigned add i64 %x, %y
	ret i64 %z
}

define i64 @sub_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = unsigned signed sub i64 %x, %y
	%z = signed unsigned sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = unsigned signed mul i64 %x, %y
	%z = signed unsigned mul i64 %x, %y
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
; CHECK: ret i64 unsigned signed add (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 signed unsigned add (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_both_ce() {
; CHECK: ret i64 unsigned signed sub (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 signed unsigned sub (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_both_ce() {
; CHECK: ret i64 unsigned signed mul (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 unsigned signed mul (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sdiv_exact_ce() {
; CHECK: ret i64 exact sdiv (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 exact sdiv (i64 ptrtoint (i64* @addr to i64), i64 91)
}
