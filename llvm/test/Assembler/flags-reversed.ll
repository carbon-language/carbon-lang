; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_both_reversed_ce() {
; CHECK: ret i64 nuw nsw add (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw nuw add (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_both_reversed_ce() {
; CHECK: ret i64 nuw nsw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw nuw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_both_reversed_ce() {
; CHECK: ret i64 nuw nsw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nsw nuw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
}
