; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_signed_ce() {
; CHECK: ret i64 nsw add (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw add (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_signed_ce() {
; CHECK: ret i64 nsw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_signed_ce() {
; CHECK: ret i64 nsw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
}
