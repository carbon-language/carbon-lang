; Test 128-bit addition in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register addition.
define void @f1(i128 *%ptr) {
; CHECK: f1:
; CHECK: algr
; CHECK: alcgr
; CHECK: br %r14
  %value = load i128 *%ptr
  %add = add i128 %value, %value
  store i128 %add, i128 *%ptr
  ret void
}

; Test memory addition with no offset.  Making the load of %a volatile
; should force the memory operand to be %b.
define void @f2(i128 *%aptr, i64 %addr) {
; CHECK: f2:
; CHECK: alg {{%r[0-5]}}, 8(%r3)
; CHECK: alcg {{%r[0-5]}}, 0(%r3)
; CHECK: br %r14
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 *%aptr
  %b = load i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the highest aligned offset that is in range of both ALG and ALCG.
define void @f3(i128 *%aptr, i64 %base) {
; CHECK: f3:
; CHECK: alg {{%r[0-5]}}, 524280(%r3)
; CHECK: alcg {{%r[0-5]}}, 524272(%r3)
; CHECK: br %r14
  %addr = add i64 %base, 524272
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 *%aptr
  %b = load i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the next doubleword up, which requires separate address logic for ALG.
define void @f4(i128 *%aptr, i64 %base) {
; CHECK: f4:
; CHECK: lgr [[BASE:%r[1-5]]], %r3
; CHECK: agfi [[BASE]], 524288
; CHECK: alg {{%r[0-5]}}, 0([[BASE]])
; CHECK: alcg {{%r[0-5]}}, 524280(%r3)
; CHECK: br %r14
  %addr = add i64 %base, 524280
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 *%aptr
  %b = load i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the next doubleword after that, which requires separate logic for
; both instructions.  It would be better to create an anchor at 524288
; that both instructions can use, but that isn't implemented yet.
define void @f5(i128 *%aptr, i64 %base) {
; CHECK: f5:
; CHECK: alg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: alcg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, 524288
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 *%aptr
  %b = load i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the lowest displacement that is in range of both ALG and ALCG.
define void @f6(i128 *%aptr, i64 %base) {
; CHECK: f6:
; CHECK: alg {{%r[0-5]}}, -524280(%r3)
; CHECK: alcg {{%r[0-5]}}, -524288(%r3)
; CHECK: br %r14
  %addr = add i64 %base, -524288
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 *%aptr
  %b = load i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

; Test the next doubleword down, which is out of range of the ALCG.
define void @f7(i128 *%aptr, i64 %base) {
; CHECK: f7:
; CHECK: alg {{%r[0-5]}}, -524288(%r3)
; CHECK: alcg {{%r[0-5]}}, 0({{%r[1-5]}})
; CHECK: br %r14
  %addr = add i64 %base, -524296
  %bptr = inttoptr i64 %addr to i128 *
  %a = load volatile i128 *%aptr
  %b = load i128 *%bptr
  %add = add i128 %a, %b
  store i128 %add, i128 *%aptr
  ret void
}

