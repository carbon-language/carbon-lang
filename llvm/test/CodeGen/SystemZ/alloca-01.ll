; Test variable-sized allocas and addresses based on them in cases where
; stack arguments are needed.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK1
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK2
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-A
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-B
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-C
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-D
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-FP

declare i64 @bar(i8 *%a, i8 *%b, i8 *%c, i8 *%d, i8 *%e, i64 %f, i64 %g)

; Allocate %length bytes and take addresses based on the result.
; There are two stack arguments, so an offset of 160 + 2 * 8 == 176
; is added to the copy of %r15.
define i64 @f1(i64 %length, i64 %index) {
; The full allocation sequence is:
;
;    la %r0, 7(%r2)      1
;    nill %r0, 0xfff8    1
;    lgr %r1, %r15         2
;    sgr %r1, %r0        1 2
;    lgr %r15, %r1         2
;
; The third instruction does not depend on the first two, so check for
; two fully-ordered sequences.
;
; FIXME: a better sequence would be:
;
;    lgr %r1, %r15
;    sgr %r1, %r2
;    nill %r1, 0xfff8
;    lgr %r15, %r1
;
; CHECK1: f1:
; CHECK1: la %r0, 7(%r2)
; CHECK1: nill %r0, 65528
; CHECK1: sgr %r1, %r0
; CHECK1: lgr %r15, %r1
;
; CHECK2: f1:
; CHECK2: lgr %r1, %r15
; CHECK2: sgr %r1, %r0
; CHECK2: lgr %r15, %r1
;
; CHECK-A: f1:
; CHECK-A: lgr %r15, %r1
; CHECK-A: la %r2, 176(%r1)
;
; CHECK-B: f1:
; CHECK-B: lgr %r15, %r1
; CHECK-B: la %r3, 177(%r1)
;
; CHECK-C: f1:
; CHECK-C: lgr %r15, %r1
; CHECK-C: la %r4, 4095({{%r3,%r1|%r1,%r3}})
;
; CHECK-D: f1:
; CHECK-D: lgr %r15, %r1
; CHECK-D: lay %r5, 4096({{%r3,%r1|%r1,%r3}})
;
; CHECK-E: f1:
; CHECK-E: lgr %r15, %r1
; CHECK-E: lay %r6, 4271({{%r3,%r1|%r1,%r3}})
;
; CHECK-FP: f1:
; CHECK-FP: lgr %r11, %r15
; CHECK-FP: lmg %r6, %r15, 224(%r11)
  %a = alloca i8, i64 %length
  %b = getelementptr i8 *%a, i64 1
  %cindex = add i64 %index, 3919
  %c = getelementptr i8 *%a, i64 %cindex
  %dindex = add i64 %index, 3920
  %d = getelementptr i8 *%a, i64 %dindex
  %eindex = add i64 %index, 4095
  %e = getelementptr i8 *%a, i64 %eindex
  %count = call i64 @bar(i8 *%a, i8 *%b, i8 *%c, i8 *%d, i8 *%e, i64 0, i64 0)
  %res = add i64 %count, 1
  ret i64 %res
}
