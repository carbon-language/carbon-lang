; RUN: opt < %s -instcombine -S | FileCheck %s

; The result of this insert is the second arg, since the top 64 bits of
; the result are undefined, and we copy the bottom 64 bits from the
; second arg
define <2 x i64> @testInsert64Bits(<2 x i64> %v, <2 x i64> %i) {
; CHECK-LABEL: @testInsert64Bits
; CHECK-NEXT: ret <2 x i64> %i
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 64, i8 0)
  ret <2 x i64> %1
}

define <2 x i64> @testZeroLength(<2 x i64> %v, <2 x i64> %i) {
; CHECK-LABEL: @testZeroLength
; CHECK-NEXT: ret <2 x i64> %i
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 0, i8 0)
  ret <2 x i64> %1
}

define <2 x i64> @testUndefinedInsertq_1(<2 x i64> %v, <2 x i64> %i) {
; CHECK-LABEL: @testUndefinedInsertq_1
; CHECK-NEXT: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 0, i8 16)
  ret <2 x i64> %1
}

define <2 x i64> @testUndefinedInsertq_2(<2 x i64> %v, <2 x i64> %i) {
; CHECK-LABEL: @testUndefinedInsertq_2
; CHECK-NEXT: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 48, i8 32)
  ret <2 x i64> %1
}

define <2 x i64> @testUndefinedInsertq_3(<2 x i64> %v, <2 x i64> %i) {
; CHECK-LABEL: @testUndefinedInsertq_3
; CHECK-NEXT: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %v, <2 x i64> %i, i8 64, i8 16)
  ret <2 x i64> %1
}

;
; Vector Demanded Bits
;

define <2 x i64> @test_extrq_arg0(<2 x i64> %x, <16 x i8> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_extrq_arg0
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %x, <16 x i8> %y)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <2 x i64> %x, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %2 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %1, <16 x i8> %y) nounwind
  ret <2 x i64> %2
}

define <2 x i64> @test_extrq_arg1(<2 x i64> %x, <16 x i8> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_extrq_arg1
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %x, <16 x i8> %y)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <16 x i8> %y, <16 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %2 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %x, <16 x i8> %1) nounwind
  ret <2 x i64> %2
}

define <2 x i64> @test_extrq_args01(<2 x i64> %x, <16 x i8> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_extrq_args01
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %x, <16 x i8> %y)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <2 x i64> %x, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %2 = shufflevector <16 x i8> %y, <16 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %3 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %1, <16 x i8> %2) nounwind
  ret <2 x i64> %3
}

define <2 x i64> @test_extrq_ret(<2 x i64> %x, <16 x i8> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_extrq_ret
; CHECK-NEXT: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %x, <16 x i8> %y) nounwind
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %2
}

define <2 x i64> @test_extrqi_arg0(<2 x i64> %x) nounwind uwtable ssp {
; CHECK-LABEL: @test_extrqi_arg0
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %x, i8 3, i8 2)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <2 x i64> %x, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %2 = tail call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %1, i8 3, i8 2)
  ret <2 x i64> %2
}

define <2 x i64> @test_extrqi_ret(<2 x i64> %x) nounwind uwtable ssp {
; CHECK-LABEL: @test_extrqi_ret
; CHECK-NEXT: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %x, i8 3, i8 2) nounwind
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %2
}

define <2 x i64> @test_insertq_arg0(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_insertq_arg0
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.insertq(<2 x i64> %x, <2 x i64> %y)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <2 x i64> %x, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertq(<2 x i64> %1, <2 x i64> %y) nounwind
  ret <2 x i64> %2
}

define <2 x i64> @test_insertq_ret(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_insertq_ret
; CHECK-NEXT: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertq(<2 x i64> %x, <2 x i64> %y) nounwind
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %2
}

define <2 x i64> @test_insertqi_arg0(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_insertqi_arg0
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %x, <2 x i64> %y, i8 3, i8 2)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <2 x i64> %x, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %y, i8 3, i8 2) nounwind
  ret <2 x i64> %2
}

define <2 x i64> @test_insertqi_arg1(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_insertqi_arg1
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %x, <2 x i64> %y, i8 3, i8 2)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <2 x i64> %y, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %2 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %x, <2 x i64> %1, i8 3, i8 2) nounwind
  ret <2 x i64> %2
}

define <2 x i64> @test_insertqi_args01(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_insertqi_args01
; CHECK-NEXT: %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %x, <2 x i64> %y, i8 3, i8 2)
; CHECK-NEXT: ret <2 x i64> %1
  %1 = shufflevector <2 x i64> %x, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %2 = shufflevector <2 x i64> %y, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  %3 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %1, <2 x i64> %2, i8 3, i8 2) nounwind
  ret <2 x i64> %3
}

define <2 x i64> @test_insertqi_ret(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK-LABEL: @test_insertqi_ret
; CHECK-NEXT: ret <2 x i64> undef
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %x, <2 x i64> %y, i8 3, i8 2) nounwind
  %2 = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %2
}

; CHECK: declare <2 x i64> @llvm.x86.sse4a.extrq
declare <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64>, <16 x i8>) nounwind

; CHECK: declare <2 x i64> @llvm.x86.sse4a.extrqi
declare <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64>, i8, i8) nounwind

; CHECK: declare <2 x i64> @llvm.x86.sse4a.insertq
declare <2 x i64> @llvm.x86.sse4a.insertq(<2 x i64>, <2 x i64>) nounwind

; CHECK: declare <2 x i64> @llvm.x86.sse4a.insertqi
declare <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64>, <2 x i64>, i8, i8) nounwind
