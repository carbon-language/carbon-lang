; Test the MSA intrinsics that are encoded with the 2R instruction format and
; convert scalars to vectors.

; RUN: llc -march=mips -mattr=+msa < %s | FileCheck %s

@llvm_mips_fill_b_ARG1 = global i32 23, align 16
@llvm_mips_fill_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_fill_b_test() nounwind {
entry:
  %0 = load i32* @llvm_mips_fill_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.fill.b(i32 %0)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_fill_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.fill.b(i32) nounwind

; CHECK: llvm_mips_fill_b_test:
; CHECK: lw
; CHECK: fill.b
; CHECK: st.b
; CHECK: .size llvm_mips_fill_b_test
;
@llvm_mips_fill_h_ARG1 = global i32 23, align 16
@llvm_mips_fill_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_fill_h_test() nounwind {
entry:
  %0 = load i32* @llvm_mips_fill_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.fill.h(i32 %0)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_fill_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.fill.h(i32) nounwind

; CHECK: llvm_mips_fill_h_test:
; CHECK: lw
; CHECK: fill.h
; CHECK: st.h
; CHECK: .size llvm_mips_fill_h_test
;
@llvm_mips_fill_w_ARG1 = global i32 23, align 16
@llvm_mips_fill_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fill_w_test() nounwind {
entry:
  %0 = load i32* @llvm_mips_fill_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.fill.w(i32 %0)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_fill_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fill.w(i32) nounwind

; CHECK: llvm_mips_fill_w_test:
; CHECK: lw
; CHECK: fill.w
; CHECK: st.w
; CHECK: .size llvm_mips_fill_w_test
;
