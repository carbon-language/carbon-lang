; RUN: opt -S -lowerbitsets < %s | FileCheck %s

target datalayout = "e-p:32:32"

; Tests that this set of globals is laid out according to our layout algorithm
; (see GlobalLayoutBuilder in include/llvm/Transforms/IPO/LowerBitSets.h).
; The chosen layout in this case is a, e, b, d, c.

; CHECK: private constant { i32, [0 x i8], i32, [0 x i8], i32, [0 x i8], i32, [0 x i8], i32 } { i32 1, [0 x i8] zeroinitializer, i32 5, [0 x i8] zeroinitializer, i32 2, [0 x i8] zeroinitializer, i32 4, [0 x i8] zeroinitializer, i32 3 }
@a = constant i32 1
@b = constant i32 2
@c = constant i32 3
@d = constant i32 4
@e = constant i32 5

!0 = !{!"bitset1", i32* @a, i32 0}
!1 = !{!"bitset1", i32* @b, i32 0}
!2 = !{!"bitset1", i32* @c, i32 0}

!3 = !{!"bitset2", i32* @b, i32 0}
!4 = !{!"bitset2", i32* @d, i32 0}

!5 = !{!"bitset3", i32* @a, i32 0}
!6 = !{!"bitset3", i32* @e, i32 0}

!llvm.bitsets = !{ !0, !1, !2, !3, !4, !5, !6 }

declare i1 @llvm.bitset.test(i8* %ptr, metadata %bitset) nounwind readnone

define void @foo() {
  %x = call i1 @llvm.bitset.test(i8* undef, metadata !"bitset1")
  %y = call i1 @llvm.bitset.test(i8* undef, metadata !"bitset2")
  %z = call i1 @llvm.bitset.test(i8* undef, metadata !"bitset3")
  ret void
}
