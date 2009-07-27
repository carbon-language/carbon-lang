; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_both_reversed_ce() {
; CHECK: ret i64 add nuw nsw (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 add nsw nuw (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_both_reversed_ce() {
; CHECK: ret i64 sub nuw nsw (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 sub nsw nuw (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_both_reversed_ce() {
; CHECK: ret i64 mul nuw nsw (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 mul nsw nuw (i64 ptrtoint (i64* @addr to i64), i64 91)
}
