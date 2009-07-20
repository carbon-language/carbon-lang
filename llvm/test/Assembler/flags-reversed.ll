; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_both_reversed_ce() {
; CHECK: ret i64 unsigned signed add (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 signed unsigned add (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_both_reversed_ce() {
; CHECK: ret i64 unsigned signed sub (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 signed unsigned sub (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_both_reversed_ce() {
; CHECK: ret i64 unsigned signed mul (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 signed unsigned mul (i64 ptrtoint (i64* @addr to i64), i64 91)
}
