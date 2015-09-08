; RUN: opt -S -lowerbitsets < %s | FileCheck %s

; Tests that we correctly create a jump table for bitsets containing 2 or more
; functions.

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-p:64:64"

; CHECK: @[[JT:.*]] = private constant [2 x <{ i8, i32, i8, i8, i8 }>] [<{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @[[FNAME:.*]] to i64), i64 ptrtoint ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)), i64 5) to i32), i8 -52, i8 -52, i8 -52 }>, <{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @[[GNAME:.*]] to i64), i64 ptrtoint ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)), i64 13) to i32), i8 -52, i8 -52, i8 -52 }>], section ".text"

; CHECK: @f = alias bitcast ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to void ()*)
; CHECK: @g = alias bitcast (<{ i8, i32, i8, i8, i8 }>* getelementptr inbounds ([2 x <{ i8, i32, i8, i8, i8 }>], [2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]], i64 0, i64 1) to void ()*)

; CHECK: define private void @[[FNAME]]() {
define void @f() {
  ret void
}

; CHECK: define private void @[[GNAME]]() {
define void @g() {
  ret void
}

!0 = !{!"bitset1", void ()* @f, i32 0}
!1 = !{!"bitset1", void ()* @g, i32 0}

!llvm.bitsets = !{ !0, !1 }

declare i1 @llvm.bitset.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  ; CHECK: sub i64 {{.*}}, ptrtoint ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)
  %x = call i1 @llvm.bitset.test(i8* %p, metadata !"bitset1")
  ret i1 %x
}
