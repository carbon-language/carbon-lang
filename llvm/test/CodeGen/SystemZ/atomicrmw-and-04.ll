; Test 64-bit atomic ANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check ANDs of a variable.
define i64 @f1(i64 %dummy, i64 *%src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: ngr %r0, %r4
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jlh [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 %b seq_cst
  ret i64 %res
}

; Check ANDs of 1, which must be done using a register.
define i64 @f2(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f2:
; CHECK: ngr
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 1 seq_cst
  ret i64 %res
}

; Check the low end of the NIHF range.
define i64 @f3(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f3:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: nihf %r0, 0
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jlh [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 4294967295 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register.
define i64 @f4(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f4:
; CHECK: ngr
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 4294967296 seq_cst
  ret i64 %res
}

; Check the low end of the NIHH range.
define i64 @f5(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f5:
; CHECK: nihh %r0, 0
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 281474976710655 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register.
define i64 @f6(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f6:
; CHECK: ngr
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 281474976710656 seq_cst
  ret i64 %res
}

; Check the highest useful NILL value.
define i64 @f7(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f7:
; CHECK: nill %r0, 65534
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -2 seq_cst
  ret i64 %res
}

; Check the low end of the NILL range.
define i64 @f8(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f8:
; CHECK: nill %r0, 0
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -65536 seq_cst
  ret i64 %res
}

; Check the highest useful NILH value, which is one less than the above.
define i64 @f9(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f9:
; CHECK: nilh %r0, 65534
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -65537 seq_cst
  ret i64 %res
}

; Check the highest useful NILF value, which is one less than the above.
define i64 @f10(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f10:
; CHECK: nilf %r0, 4294901758
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -65538 seq_cst
  ret i64 %res
}

; Check the low end of the NILH range.
define i64 @f11(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f11:
; CHECK: nilh %r0, 0
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -4294901761 seq_cst
  ret i64 %res
}

; Check the low end of the NILF range.
define i64 @f12(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f12:
; CHECK: nilf %r0, 0
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -4294967296 seq_cst
  ret i64 %res
}

; Check the highest useful NIHL value, which is one less than the above.
define i64 @f13(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f13:
; CHECK: nihl %r0, 65534
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -4294967297 seq_cst
  ret i64 %res
}

; Check the low end of the NIHL range.
define i64 @f14(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f14:
; CHECK: nihl %r0, 0
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -281470681743361 seq_cst
  ret i64 %res
}

; Check the highest useful NIHH value, which is 1<<32 less than the above.
define i64 @f15(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f15:
; CHECK: nihh %r0, 65534
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -281474976710657 seq_cst
  ret i64 %res
}

; Check the highest useful NIHF value, which is 1<<32 less than the above.
define i64 @f16(i64 %dummy, i64 *%src) {
; CHECK-LABEL: f16:
; CHECK: nihf %r0, 4294901758
; CHECK: br %r14
  %res = atomicrmw and i64 *%src, i64 -281479271677953 seq_cst
  ret i64 %res
}
