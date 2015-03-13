; RUN: opt -S -lowerbitsets < %s | FileCheck %s

target datalayout = "e-p:32:32"

@a = constant i32 1
@b = constant [2 x i32] [i32 2, i32 3]

!0 = !{!"bitset1", i32* @a, i32 0}
!1 = !{!"bitset1", [2 x i32]* @b, i32 4}

!llvm.bitsets = !{ !0, !1 }

declare i1 @llvm.bitset.test(i8* %ptr, metadata %bitset) nounwind readnone

; CHECK: @foo(
define i1 @foo() {
  ; CHECK: ret i1 true
  %x = call i1 @llvm.bitset.test(i8* bitcast (i32* @a to i8*), metadata !"bitset1")
  ret i1 %x
}

; CHECK: @bar(
define i1 @bar() {
  ; CHECK: ret i1 true
  %x = call i1 @llvm.bitset.test(i8* bitcast (i32* getelementptr ([2 x i32], [2 x i32]* @b, i32 0, i32 1) to i8*), metadata !"bitset1")
  ret i1 %x
}

; CHECK: @baz(
define i1 @baz() {
  ; CHECK-NOT: ret i1 true
  %x = call i1 @llvm.bitset.test(i8* bitcast (i32* getelementptr ([2 x i32], [2 x i32]* @b, i32 0, i32 0) to i8*), metadata !"bitset1")
  ret i1 %x
}
