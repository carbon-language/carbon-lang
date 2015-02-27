; Test the MSA intrinsics that are encoded with the I10 instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

@llvm_mips_bnz_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16

define i32 @llvm_mips_bnz_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bnz_b_ARG1
  %1 = tail call i32 @llvm.mips.bnz.b(<16 x i8> %0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %true, label %false
true:
  ret i32 2
false:
  ret i32 3
}

declare i32 @llvm.mips.bnz.b(<16 x i8>) nounwind

; CHECK: llvm_mips_bnz_b_test:
; CHECK-DAG: ld.b [[R0:\$w[0-9]+]]
; CHECK-DAG: bnz.b [[R0]]
; CHECK: .size llvm_mips_bnz_b_test

@llvm_mips_bnz_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16

define i32 @llvm_mips_bnz_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bnz_h_ARG1
  %1 = tail call i32 @llvm.mips.bnz.h(<8 x i16> %0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %true, label %false
true:
  ret i32 2
false:
  ret i32 3
}

declare i32 @llvm.mips.bnz.h(<8 x i16>) nounwind

; CHECK: llvm_mips_bnz_h_test:
; CHECK-DAG: ld.h [[R0:\$w[0-9]+]]
; CHECK-DAG: bnz.h [[R0]]
; CHECK: .size llvm_mips_bnz_h_test

@llvm_mips_bnz_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16

define i32 @llvm_mips_bnz_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bnz_w_ARG1
  %1 = tail call i32 @llvm.mips.bnz.w(<4 x i32> %0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %true, label %false
true:
  ret i32 2
false:
  ret i32 3
}

declare i32 @llvm.mips.bnz.w(<4 x i32>) nounwind

; CHECK: llvm_mips_bnz_w_test:
; CHECK-DAG: ld.w [[R0:\$w[0-9]+]]
; CHECK-DAG: bnz.w [[R0]]
; CHECK: .size llvm_mips_bnz_w_test

@llvm_mips_bnz_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16

define i32 @llvm_mips_bnz_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bnz_d_ARG1
  %1 = tail call i32 @llvm.mips.bnz.d(<2 x i64> %0)
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %true, label %false
true:
  ret i32 2
false:
  ret i32 3
}

declare i32 @llvm.mips.bnz.d(<2 x i64>) nounwind

; CHECK: llvm_mips_bnz_d_test:
; CHECK-DAG: ld.d [[R0:\$w[0-9]+]]
; CHECK-DAG: bnz.d [[R0]]
; CHECK: .size llvm_mips_bnz_d_test

