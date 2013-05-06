; Test 32-bit byteswaps from memory to registers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.bswap.i32(i32 %a)

; Check LRV with no displacement.
define i32 @f1(i32 *%src) {
; CHECK: f1:
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %a = load i32 *%src
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the high end of the aligned LRV range.
define i32 @f2(i32 *%src) {
; CHECK: f2:
; CHECK: lrv %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131071
  %a = load i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f3(i32 *%src) {
; CHECK: f3:
; CHECK: agfi %r2, 524288
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131072
  %a = load i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the high end of the negative aligned LRV range.
define i32 @f4(i32 *%src) {
; CHECK: f4:
; CHECK: lrv %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -1
  %a = load i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the low end of the LRV range.
define i32 @f5(i32 *%src) {
; CHECK: f5:
; CHECK: lrv %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131072
  %a = load i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i32 *%src) {
; CHECK: f6:
; CHECK: agfi %r2, -524292
; CHECK: lrv %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131073
  %a = load i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}

; Check that LRV allows an index.
define i32 @f7(i64 %src, i64 %index) {
; CHECK: f7:
; CHECK: lrv %r2, 524287({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %a = load i32 *%ptr
  %swapped = call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %swapped
}
