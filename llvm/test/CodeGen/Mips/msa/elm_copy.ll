; Test the MSA intrinsics that are encoded with the ELM instruction format and
; are element extraction operations.

; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

@llvm_mips_copy_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_copy_s_b_RES  = global i32 0, align 16

define void @llvm_mips_copy_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_copy_s_b_ARG1
  %1 = tail call i32 @llvm.mips.copy.s.b(<16 x i8> %0, i32 1)
  store i32 %1, i32* @llvm_mips_copy_s_b_RES
  ret void
}

declare i32 @llvm.mips.copy.s.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_copy_s_b_test:
; CHECK: ld.b
; CHECK: copy_s.b
; CHECK: sw
; CHECK: .size llvm_mips_copy_s_b_test
;
@llvm_mips_copy_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_copy_s_h_RES  = global i32 0, align 16

define void @llvm_mips_copy_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_copy_s_h_ARG1
  %1 = tail call i32 @llvm.mips.copy.s.h(<8 x i16> %0, i32 1)
  store i32 %1, i32* @llvm_mips_copy_s_h_RES
  ret void
}

declare i32 @llvm.mips.copy.s.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_copy_s_h_test:
; CHECK: ld.h
; CHECK: copy_s.h
; CHECK: sw
; CHECK: .size llvm_mips_copy_s_h_test
;
@llvm_mips_copy_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_copy_s_w_RES  = global i32 0, align 16

define void @llvm_mips_copy_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_copy_s_w_ARG1
  %1 = tail call i32 @llvm.mips.copy.s.w(<4 x i32> %0, i32 1)
  store i32 %1, i32* @llvm_mips_copy_s_w_RES
  ret void
}

declare i32 @llvm.mips.copy.s.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_copy_s_w_test:
; CHECK: ld.w
; CHECK: copy_s.w
; CHECK: sw
; CHECK: .size llvm_mips_copy_s_w_test
;
@llvm_mips_copy_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_copy_s_d_RES  = global i64 0, align 16

define void @llvm_mips_copy_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_copy_s_d_ARG1
  %1 = tail call i64 @llvm.mips.copy.s.d(<2 x i64> %0, i32 1)
  store i64 %1, i64* @llvm_mips_copy_s_d_RES
  ret void
}

declare i64 @llvm.mips.copy.s.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_copy_s_d_test:
; CHECK: ld.w
; CHECK: copy_s.w
; CHECK: copy_s.w
; CHECK: sw
; CHECK: sw
; CHECK: .size llvm_mips_copy_s_d_test
;
@llvm_mips_copy_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_copy_u_b_RES  = global i32 0, align 16

define void @llvm_mips_copy_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>* @llvm_mips_copy_u_b_ARG1
  %1 = tail call i32 @llvm.mips.copy.u.b(<16 x i8> %0, i32 1)
  store i32 %1, i32* @llvm_mips_copy_u_b_RES
  ret void
}

declare i32 @llvm.mips.copy.u.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_copy_u_b_test:
; CHECK: ld.b
; CHECK: copy_u.b
; CHECK: sw
; CHECK: .size llvm_mips_copy_u_b_test
;
@llvm_mips_copy_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_copy_u_h_RES  = global i32 0, align 16

define void @llvm_mips_copy_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>* @llvm_mips_copy_u_h_ARG1
  %1 = tail call i32 @llvm.mips.copy.u.h(<8 x i16> %0, i32 1)
  store i32 %1, i32* @llvm_mips_copy_u_h_RES
  ret void
}

declare i32 @llvm.mips.copy.u.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_copy_u_h_test:
; CHECK: ld.h
; CHECK: copy_u.h
; CHECK: sw
; CHECK: .size llvm_mips_copy_u_h_test
;
@llvm_mips_copy_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_copy_u_w_RES  = global i32 0, align 16

define void @llvm_mips_copy_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>* @llvm_mips_copy_u_w_ARG1
  %1 = tail call i32 @llvm.mips.copy.u.w(<4 x i32> %0, i32 1)
  store i32 %1, i32* @llvm_mips_copy_u_w_RES
  ret void
}

declare i32 @llvm.mips.copy.u.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_copy_u_w_test:
; CHECK: ld.w
; CHECK: copy_u.w
; CHECK: sw
; CHECK: .size llvm_mips_copy_u_w_test
;
@llvm_mips_copy_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_copy_u_d_RES  = global i64 0, align 16

define void @llvm_mips_copy_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>* @llvm_mips_copy_u_d_ARG1
  %1 = tail call i64 @llvm.mips.copy.u.d(<2 x i64> %0, i32 1)
  store i64 %1, i64* @llvm_mips_copy_u_d_RES
  ret void
}

declare i64 @llvm.mips.copy.u.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_copy_u_d_test:
; CHECK: ld.w
; CHECK: copy_s.w
; CHECK: copy_s.w
; CHECK: sw
; CHECK: sw
; CHECK: .size llvm_mips_copy_u_d_test
;
