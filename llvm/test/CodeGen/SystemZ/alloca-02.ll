; Make sure that the alloca offset isn't lost when the alloca result is
; used directly in a load or store.  There must always be an LA or LAY.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-A
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-B
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-C
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-D

declare i64 @bar(i8 *%a)

define i64 @f1(i64 %length, i64 %index) {
; CHECK-A: f1:
; CHECK-A: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-A: la %r2, 160([[ADDR]])
; CHECK-A: mvi 0(%r2), 0
;
; CHECK-B: f1:
; CHECK-B: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-B: la %r2, 160([[ADDR]])
; CHECK-B: mvi 4095(%r2), 1
;
; CHECK-C: f1:
; CHECK-C: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-C: la [[TMP:%r[1-5]]], 160(%r3,[[ADDR]])
; CHECK-C: mvi 0([[TMP]]), 2
;
; CHECK-D: f1:
; CHECK-D: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-D: la [[TMP:%r[1-5]]], 160(%r3,[[ADDR]])
; CHECK-D: mvi 4095([[TMP]]), 3
;
; CHECK-E: f1:
; CHECK-E: lgr %r15, [[ADDR:%r[1-5]]]
; CHECK-E: la [[TMP:%r[1-5]]], 160(%r3,[[ADDR]])
; CHECK-E: mviy 4096([[TMP]]), 4
  %a = alloca i8, i64 %length
  store i8 0, i8 *%a
  %b = getelementptr i8 *%a, i64 4095
  store i8 1, i8 *%b
  %c = getelementptr i8 *%a, i64 %index
  store i8 2, i8 *%c
  %d = getelementptr i8 *%c, i64 4095
  store i8 3, i8 *%d
  %e = getelementptr i8 *%d, i64 1
  store i8 4, i8 *%e
  %count = call i64 @bar(i8 *%a)
  %res = add i64 %count, 1
  ret i64 %res
}
