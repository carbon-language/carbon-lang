; Test strlen using SRST, i32 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @strlen(i8 *%src)
declare i32 @strnlen(i8 *%src, i32 %len)

; Test strlen with an i32-based prototype.  It would also be valid for
; the uses of %r3 and REG after the LGR to be swapped.
define i32 @f1(i32 %dummy, i8 *%src) {
; CHECK-LABEL: f1:
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: lghi %r2, 0
; CHECK-DAG: lgr [[REG:%r[145]]], %r3
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: srst %r2, [[REG]]
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NEXT: BB#{{[0-9]+}}
; CHECK-NEXT: sgr %r2, %r3
; CHECK: br %r14
  %res = call i32 @strlen(i8 *%src)
  ret i32 %res
}

; Test strnlen with an i32-based prototype.
define i32 @f2(i32 zeroext %len, i8 *%src) {
; CHECK-LABEL: f2:
; CHECK-DAG: agr %r2, %r3
; CHECK-DAG: lhi %r0, 0
; CHECK-DAG: lgr [[REG:%r[145]]], %r3
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK-NEXT: srst %r2, [[REG]]
; CHECK-NEXT: jo [[LABEL]]
; CHECK-NEXT: BB#{{[0-9]+}}
; CHECK-NEXT: sgr %r2, %r3
; CHECK: br %r14
  %res = call i32 @strnlen(i8 *%src, i32 %len)
  ret i32 %res
}
