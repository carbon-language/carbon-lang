; RUN: opt < %s -instcombine -S | FileCheck %s

; We should optimize these two redundant insertqi into one
; CHECK: define <2 x i64> @testInsertTwice(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertTwice(<2 x i64> %v, <2 x i64> %i) {
; CHECK: call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 32)
; CHECK-NOT: insertqi
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 32)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 32, i8 32)
  ret <2 x i64> %2
}

; The result of this insert is the second arg, since the top 64 bits of
; the result are undefined, and we copy the bottom 64 bits from the
; second arg
; CHECK: define <2 x i64> @testInsert64Bits(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsert64Bits(<2 x i64> %v, <2 x i64> %i) {
; CHECK: ret <2 x i64> %i
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 64, i8 0)
  ret <2 x i64> %1
}

; Test the several types of ranges and ordering that exist for two insertqi
; CHECK: define <2 x i64> @testInsertContainedRange(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertContainedRange(<2 x i64> %v, <2 x i64> %i) {
; CHECK: %[[RES:.*]] = call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 0)
; CHECK: ret <2 x i64> %[[RES]]
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 0)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 16, i8 16)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testInsertContainedRange_2(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertContainedRange_2(<2 x i64> %v, <2 x i64> %i) {
; CHECK: %[[RES:.*]] = call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 0)
; CHECK: ret <2 x i64> %[[RES]]
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 16, i8 16)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 32, i8 0)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testInsertOverlappingRange(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertOverlappingRange(<2 x i64> %v, <2 x i64> %i) {
; CHECK: %[[RES:.*]] = call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 48, i8 0)
; CHECK: ret <2 x i64> %[[RES]]
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 0)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 32, i8 16)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testInsertOverlappingRange_2(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertOverlappingRange_2(<2 x i64> %v, <2 x i64> %i) {
; CHECK: %[[RES:.*]] = call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 48, i8 0)
; CHECK: ret <2 x i64> %[[RES]]
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 16)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 32, i8 0)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testInsertAdjacentRange(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertAdjacentRange(<2 x i64> %v, <2 x i64> %i) {
; CHECK: %[[RES:.*]] = call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 48, i8 0)
; CHECK: ret <2 x i64> %[[RES]]
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 32, i8 0)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 16, i8 32)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testInsertAdjacentRange_2(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertAdjacentRange_2(<2 x i64> %v, <2 x i64> %i) {
; CHECK: %[[RES:.*]] = call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 48, i8 0)
; CHECK: ret <2 x i64> %[[RES]]
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 16, i8 32)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 32, i8 0)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testInsertDisjointRange(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertDisjointRange(<2 x i64> %v, <2 x i64> %i) {
; CHECK: tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 16, i8 0)
; CHECK: tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 16, i8 32)
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 16, i8 0)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 16, i8 32)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testInsertDisjointRange_2(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testInsertDisjointRange_2(<2 x i64> %v, <2 x i64> %i) {
; CHECK:  tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 16, i8 0)
; CHECK:  tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 16, i8 32)
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 16, i8 0)
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %i, i8 16, i8 32)
  ret <2 x i64> %2
}

; CHECK: define <2 x i64> @testZeroLength(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testZeroLength(<2 x i64> %v, <2 x i64> %i) {
; CHECK: ret <2 x i64> %i
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 0, i8 0)
  ret <2 x i64> %1
}

; CHECK: define <2 x i64> @testUndefinedInsertq_1(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testUndefinedInsertq_1(<2 x i64> %v, <2 x i64> %i) {
; CHECK: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 0, i8 16)
  ret <2 x i64> %1
}

; CHECK: define <2 x i64> @testUndefinedInsertq_2(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testUndefinedInsertq_2(<2 x i64> %v, <2 x i64> %i) {
; CHECK: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 48, i8 32)
  ret <2 x i64> %1
}

; CHECK: define <2 x i64> @testUndefinedInsertq_3(<2 x i64> %v, <2 x i64> %i)
define <2 x i64> @testUndefinedInsertq_3(<2 x i64> %v, <2 x i64> %i) {
; CHECK: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 64, i8 16)
  ret <2 x i64> %1
}

; CHECK: declare <2 x i64> @llvm.x86.sse4a.insertqi
declare <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64>, <2 x i64>, i8, i8) nounwind
