; Test memcmp using CLC.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare signext i32 @memcmp(i8 *%src1, i8 *%src2, i64 %size)

; Zero-length comparisons should be optimized away.
define i32 @f1(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f1:
; CHECK: lhi %r2, 0
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 0)
  ret i32 %res
}

; Check a case where the result is used as an integer.
define i32 @f2(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f2:
; CHECK: clc 0(2,%r2), 0(%r3)
; CHECK: ipm %r2
; CHECK: sll %r2, 2
; CHECK: sra %r2, 30
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 2)
  ret i32 %res
}

; Check a case where the result is tested for equality.
define void @f3(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f3:
; CHECK: clc 0(3,%r2), 0(%r3)
; CHECK-NEXT: je {{\..*}}
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 3)
  %cmp = icmp eq i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for inequality.
define void @f4(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f4:
; CHECK: clc 0(4,%r2), 0(%r3)
; CHECK-NEXT: jlh {{\..*}}
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 4)
  %cmp = icmp ne i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested via slt.
define void @f5(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f5:
; CHECK: clc 0(5,%r2), 0(%r3)
; CHECK-NEXT: jl {{\..*}}
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 5)
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check a case where the result is tested for sgt.
define void @f6(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f6:
; CHECK: clc 0(6,%r2), 0(%r3)
; CHECK-NEXT: jh {{\..*}}
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 6)
  %cmp = icmp sgt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret void
}

; Check the upper end of the CLC range.  Here the result is used both as
; an integer and for branching, but it's better to branch on the result
; of the SRA.
define i32 @f7(i8 *%src1, i8 *%src2, i32 *%dest) {
; CHECK-LABEL: f7:
; CHECK: clc 0(256,%r2), 0(%r3)
; CHECK: ipm %r2
; CHECK: sll %r2, 2
; CHECK: sra %r2, 30
; CHECK: jl {{.L*}}
; CHECK: br %r14
entry:
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 256)
  %cmp = icmp slt i32 %res, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 0, i32 *%dest
  br label %exit

exit:
  ret i32 %res
}

; 257 bytes is too big for a single CLC.  For now expect a call instead.
define i32 @f8(i8 *%src1, i8 *%src2) {
; CHECK-LABEL: f8:
; CHECK: brasl %r14, memcmp@PLT
; CHECK: br %r14
  %res = call i32 @memcmp(i8 *%src1, i8 *%src2, i64 257)
  ret i32 %res
}
