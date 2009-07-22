; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_unsigned_ce() {
; CHECK: ret i64 nuw add (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nuw add (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_unsigned_ce() {
; CHECK: ret i64 nuw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nuw sub (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_unsigned_ce() {
; CHECK: ret i64 nuw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 nuw mul (i64 ptrtoint (i64* @addr to i64), i64 91)
}
