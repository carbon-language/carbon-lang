; Test the MSA intrinsics that are encoded with the I5 instruction format and
; are loads or stores.

; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

@llvm_mips_ld_b_ARG = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ld_b_RES = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ld_b_test() nounwind {
entry:
  %0 = bitcast <16 x i8>* @llvm_mips_ld_b_ARG to i8*
  %1 = tail call <16 x i8> @llvm.mips.ld.b(i8* %0, i32 16)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_ld_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ld.b(i8*, i32) nounwind

; CHECK: llvm_mips_ld_b_test:
; CHECK: ld.b [[R1:\$w[0-9]+]], 16(
; CHECK: st.b
; CHECK: .size llvm_mips_ld_b_test
;
@llvm_mips_ld_h_ARG = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ld_h_RES = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ld_h_test() nounwind {
entry:
  %0 = bitcast <8 x i16>* @llvm_mips_ld_h_ARG to i8*
  %1 = tail call <8 x i16> @llvm.mips.ld.h(i8* %0, i32 16)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_ld_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ld.h(i8*, i32) nounwind

; CHECK: llvm_mips_ld_h_test:
; CHECK: ld.h [[R1:\$w[0-9]+]], 16(
; CHECK: st.h
; CHECK: .size llvm_mips_ld_h_test
;
@llvm_mips_ld_w_ARG = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ld_w_RES = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ld_w_test() nounwind {
entry:
  %0 = bitcast <4 x i32>* @llvm_mips_ld_w_ARG to i8*
  %1 = tail call <4 x i32> @llvm.mips.ld.w(i8* %0, i32 16)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ld_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ld.w(i8*, i32) nounwind

; CHECK: llvm_mips_ld_w_test:
; CHECK: ld.w [[R1:\$w[0-9]+]], 16(
; CHECK: st.w
; CHECK: .size llvm_mips_ld_w_test
;
@llvm_mips_ld_d_ARG = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ld_d_RES = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ld_d_test() nounwind {
entry:
  %0 = bitcast <2 x i64>* @llvm_mips_ld_d_ARG to i8*
  %1 = tail call <2 x i64> @llvm.mips.ld.d(i8* %0, i32 16)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ld_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ld.d(i8*, i32) nounwind

; CHECK: llvm_mips_ld_d_test:
; CHECK: ld.d [[R1:\$w[0-9]+]], 16(
; CHECK: st.d
; CHECK: .size llvm_mips_ld_d_test
;
@llvm_mips_st_b_ARG = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_st_b_RES = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_st_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_st_b_ARG
  %1 = bitcast <16 x i8>* @llvm_mips_st_b_RES to i8*
  tail call void @llvm.mips.st.b(<16 x i8> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.b(<16 x i8>, i8*, i32) nounwind

; CHECK: llvm_mips_st_b_test:
; CHECK: ld.b
; CHECK: st.b [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_b_test
;
@llvm_mips_st_h_ARG = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_st_h_RES = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_st_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_st_h_ARG
  %1 = bitcast <8 x i16>* @llvm_mips_st_h_RES to i8*
  tail call void @llvm.mips.st.h(<8 x i16> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.h(<8 x i16>, i8*, i32) nounwind

; CHECK: llvm_mips_st_h_test:
; CHECK: ld.h
; CHECK: st.h [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_h_test
;
@llvm_mips_st_w_ARG = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_st_w_RES = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_st_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_st_w_ARG
  %1 = bitcast <4 x i32>* @llvm_mips_st_w_RES to i8*
  tail call void @llvm.mips.st.w(<4 x i32> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.w(<4 x i32>, i8*, i32) nounwind

; CHECK: llvm_mips_st_w_test:
; CHECK: ld.w
; CHECK: st.w [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_w_test
;
@llvm_mips_st_d_ARG = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_st_d_RES = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_st_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_st_d_ARG
  %1 = bitcast <2 x i64>* @llvm_mips_st_d_RES to i8*
  tail call void @llvm.mips.st.d(<2 x i64> %0, i8* %1, i32 16)
  ret void
}

declare void @llvm.mips.st.d(<2 x i64>, i8*, i32) nounwind

; CHECK: llvm_mips_st_d_test:
; CHECK: ld.d
; CHECK: st.d [[R1:\$w[0-9]+]], 16(
; CHECK: .size llvm_mips_st_d_test
;
