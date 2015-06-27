; RUN: opt -S -lowerbitsets < %s | FileCheck %s

target datalayout = "e-p:32:32"

; CHECK-NOT: @b = alias
@a = constant i32 1
@b = constant [2 x i32] [i32 2, i32 3]

!0 = !{!"bitset1", i32* @a, i32 0}
!1 = !{!"bitset1", i32* bitcast ([2 x i32]* @b to i32*), i32 0}

!llvm.bitsets = !{ !0, !1 }

declare i1 @llvm.bitset.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  %x = call i1 @llvm.bitset.test(i8* %p, metadata !"bitset1")
  ret i1 %x
}
