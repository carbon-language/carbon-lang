; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@addr = external global i64

define i64 @add_plain_ce() {
; CHECK: ret i64 add (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 add (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sub_plain_ce() {
; CHECK: ret i64 sub (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 sub (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @mul_plain_ce() {
; CHECK: ret i64 mul (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 mul (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64 @sdiv_plain_ce() {
; CHECK: ret i64 sdiv (i64 ptrtoint (i64* @addr to i64), i64 91)
	ret i64 sdiv (i64 ptrtoint (i64* @addr to i64), i64 91)
}

define i64* @gep_plain_ce() {
; CHECK: ret i64* getelementptr (i64* @addr, i64 171)
        ret i64* getelementptr (i64* @addr, i64 171)
}
