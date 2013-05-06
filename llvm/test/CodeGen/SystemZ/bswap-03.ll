; Test 64-bit byteswaps from memory to registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @llvm.bswap.i64(i64 %a)

; Check LRVG with no displacement.
define i64 @f1(i64 *%src) {
; CHECK: f1:
; CHECK: lrvg %r2, 0(%r2)
; CHECK: br %r14
  %a = load i64 *%src
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the high end of the aligned LRVG range.
define i64 @f2(i64 *%src) {
; CHECK: f2:
; CHECK: lrvg %r2, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65535
  %a = load i64 *%ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f3(i64 *%src) {
; CHECK: f3:
; CHECK: agfi %r2, 524288
; CHECK: lrvg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65536
  %a = load i64 *%ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the high end of the negative aligned LRVG range.
define i64 @f4(i64 *%src) {
; CHECK: f4:
; CHECK: lrvg %r2, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -1
  %a = load i64 *%ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the low end of the LRVG range.
define i64 @f5(i64 *%src) {
; CHECK: f5:
; CHECK: lrvg %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65536
  %a = load i64 *%ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i64 *%src) {
; CHECK: f6:
; CHECK: agfi %r2, -524296
; CHECK: lrvg %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65537
  %a = load i64 *%ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}

; Check that LRVG allows an index.
define i64 @f7(i64 %src, i64 %index) {
; CHECK: f7:
; CHECK: lrvg %r2, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %a = load i64 *%ptr
  %swapped = call i64 @llvm.bswap.i64(i64 %a)
  ret i64 %swapped
}
